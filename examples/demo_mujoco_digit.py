# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import sys
import time

import cv2
import hydra
from omegaconf import OmegaConf
import mujoco as mj
import pyrender
import tacto  # Import TACTO
from mujoco.viewer import launch
import glfw
import threading
from math import pi
import numpy as np 
np.set_printoptions(precision=3, suppress=True)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CONTROL_INCREMENT = 0.0005  # Amount to move joints per keypress

SCENE_ROOT = os.path.join(CURRENT_DIR, "xml")
SCENE_FILE = os.path.join(SCENE_ROOT, "touch_playground_rot.xml")
TACTO_DIR = os.path.join(SCENE_ROOT, "assets", "tacto_conf")
CAN_XYZ_LIM = 0.008

def key_callback(window, key, scancode, action, mods):
    """
    GLFW key callback to interactively control joint positions.
    """
    if action != glfw.PRESS and action != glfw.REPEAT:
        return  # Ignore key releases

    elif key == glfw.KEY_R:
        data.qpos[model.joint("xx").qposadr] = min(data.qpos[model.joint("xx").qposadr] + CONTROL_INCREMENT, CAN_XYZ_LIM)
    elif key == glfw.KEY_F:
        data.qpos[model.joint("xx").qposadr] = max(data.qpos[model.joint("xx").qposadr] - CONTROL_INCREMENT, -CAN_XYZ_LIM)
    elif key == glfw.KEY_W:
        data.qpos[model.joint("yy").qposadr] = min(data.qpos[model.joint("yy").qposadr] + CONTROL_INCREMENT, CAN_XYZ_LIM)
    elif key == glfw.KEY_S:
        data.qpos[model.joint("yy").qposadr] = max(data.qpos[model.joint("yy").qposadr] - CONTROL_INCREMENT, -CAN_XYZ_LIM)
    elif key == glfw.KEY_D:
        data.qpos[model.joint("zz").qposadr] = min(data.qpos[model.joint("zz").qposadr] + CONTROL_INCREMENT, CAN_XYZ_LIM)
    elif key == glfw.KEY_A:
        data.qpos[model.joint("zz").qposadr] = max(data.qpos[model.joint("zz").qposadr] - CONTROL_INCREMENT, -CAN_XYZ_LIM)
    elif key == glfw.KEY_J:
        data.qpos[model.joint("master_joint_x").qposadr] += (pi/20)
    elif key == glfw.KEY_K:
        data.qpos[model.joint("master_joint_y").qposadr] += (pi/20)
    elif key == glfw.KEY_L:
        data.qpos[model.joint("master_joint_z").qposadr] += (pi/20)


log = logging.getLogger(__name__)

def main(cfg):
    
    # For allowing keyboard input in the viewer
    global model, data, camera, scene
    
    #--------------#
    # MuJoCo Setup #
    #--------------#
    model = mj.MjModel.from_xml_path(
        str(SCENE_FILE)
    )
    data = mj.MjData(model)
    model.opt.timestep = 0.001
    mj.mj_step(model, data) # step to initialize object poses
    

    #--------------------#
    # TACTO Sensor Setup #
    #--------------------#
    #   Initialize tactos - this needs to be done before creating the glfw window,
    # probably due to the way glfw contexts are managed internally. 
    bg = cv2.imread(os.path.join(TACTO_DIR, "bg_digit_240_320.jpg"))
    tactos = tacto.Sensor(**cfg.tacto, background=bg)


    #   Add the objects from MuJoCo to the Tacto sensor
    # Generally, you should use the add_geom_mujoco, unless you are 100% sure that the geom and the body is aligned.
    # More often than not, that is not the case.
    # Here, can_geom is the name of the geom defined in the touch_playground_rot.xml file.
    # can_mesh is optional, but if not provided, the sensor will try to look for a mesh with the name "can_geom"+"_mesh"
    tactos.add_geom_mujoco("can_geom", model, data, "can_mesh")

    #   The function for adding body is there too.
    # tactos.add_body_mujoco("can", model, data, "can_mesh")
    
    #   Now we add the pyrender camera to the tacto sensor.
    # The name of its location is the site where the touch sensor is located in the xml file.
    # In this case, it is "left_tacto_pad".
    tactos.add_camera_mujoco(
        "left_tacto_pad", model, data
    )

    #   You can add more than 1 camera so long as the appropriate sites and sensors are defined in the xml file.
    # tactos.add_camera_mujoco(
    #     "right_tacto_pad", model, data
    # )
    
    #----------------------------#
    # MuJoCo Visualization Setup #
    #----------------------------#
    # Create GLFW window
    if not glfw.init():
       raise Exception("Failed to initialize GLFW")
    window = glfw.create_window(1200, 800, "MuJoCo Interactive Viewer", None, None)
    if not window:
        glfw.terminate()
        raise RuntimeError("GLFW window creation failed")

    glfw.make_context_current(window)
    glfw.set_key_callback(window, key_callback)
    options = mj.MjvOption()
    scene = mj.MjvScene(model, maxgeom=1000)
    context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150)
    camera = mj.MjvCamera()
    camera.type = mj.mjtCamera.mjCAMERA_TRACKING
    camera.trackbodyid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "master")

    #   You can choose to render the sensor in a separate thread.
    # Otherwise, you can include the tactos.render(model, data) in the main MuJoCo loop.
    def render_digit():
        while not glfw.window_should_close(window):
            # This will render the color and depth images of all tacto sensors.
            # You get an array of color and depth images, one for each sensor.
            color, depth = tactos.render(model, data)
            # The GUI automatically renders all color and depth images from multiple sensors side-by-side.
            tactos.updateGUI(color, depth)
            time.sleep(0.01)
    render_thread = threading.Thread(target=render_digit)
    render_thread.start()

    while not glfw.window_should_close(window):
        # step simulation of mujoco model
        data.qpos[model.joint("xx").qposadr] = min(data.qpos[model.joint("xx").qposadr], -0.001)
        data.qpos[model.joint("yy").qposadr] = max(min(data.qpos[model.joint("yy").qposadr], CAN_XYZ_LIM), -CAN_XYZ_LIM)
        data.qpos[model.joint("zz").qposadr] = max(min(data.qpos[model.joint("zz").qposadr], CAN_XYZ_LIM), -CAN_XYZ_LIM)
        mj.mj_step(model, data)

        #   If you don't want to use the separate render thread, 
        # You could render the tacto sensor in the main MuJoCo loop. 
        # Just uncomment the following lines (after commenting out the render_thread).
        # In practice, you would want to render the tacto sensor according to some frame rate condition.
        # color, depth = tactos.render(model, data)
        # tactos.updateGUI(color, depth)

        # visualize the simulation
        # Render the scene
        viewport_width, viewport_height = glfw.get_framebuffer_size(window)
        mj.mjv_updateScene(
            model, data, options, None, camera, mj.mjtCatBit.mjCAT_ALL.value, scene
        )
        mj.mjr_render(mj.MjrRect(0, 0, viewport_width, viewport_height), scene, context)

        glfw.swap_buffers(window)
        glfw.poll_events()

if __name__ == "__main__":
    hydra.initialize("./xml/assets/tacto_conf", version_base=None)
    script_dir = CURRENT_DIR
    cfg = hydra.compose("digit.yaml", overrides=[f"base_dir={script_dir}"])
    main(cfg)
