# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import sys
import collections
import logging
import os
import warnings
from dataclasses import dataclass

import cv2
import numpy as np
import pybullet as p
import trimesh
from urdfpy import URDF

from .renderer import Renderer
import pyrender
import mujoco as mj
from scipy.spatial.transform import Rotation as R

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _get_default_config(filename):
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), filename)


def get_digit_config_path():
    return _get_default_config("config_digit.yml")


def get_digit_shadow_config_path():
    return _get_default_config("config_digit_shadow.yml")


def get_omnitact_config_path():
    return _get_default_config("config_omnitact.yml")

# Function to create a coordinate frame mesh in pyrender
def create_coordinate_frame_mesh(size=2):
    """Create a mesh representing the X, Y, and Z axes."""
    # Define the lines for the X, Y, and Z axes
    vertices = np.array([
        [0, 0, 0],  # Origin
        [size, 0, 0],  # X-axis
        [0, size, 0],  # Y-axis
        [0, 0, size]   # Z-axis
    ])
    # Define the indices for the lines
    indices = np.array([
        [0, 1],  # X-axis
        [0, 2],  # Y-axis
        [0, 3]   # Z-axis
    ])
    # Define colors for each axis (RGB)
    colors = np.array([
        [255, 0, 0],  # Red for X-axis
        [0, 255, 0],  # Green for Y-axis
        [0, 0, 255]   # Blue for Z-axis
    ])
    # Create a mesh from the lines
    # Create a primitive for each axis
    primitives = []
    for i, (start, end) in enumerate(indices):
        primitives.append(
            pyrender.Primitive(
                positions=vertices[[start, end]],
                color_0=colors[i],
                mode=pyrender.constants.GLTF.LINES
            )
        )
    # Create a mesh from the primitives
    return pyrender.Mesh(primitives=primitives, is_visible=True)

@dataclass
class Link:
    obj_id: int  # pybullet ID
    link_id: int  # pybullet link ID (-1 means base)
    cid: int  # physicsClientId
    mujoco_data: any = None
    mujoco_model: any = None
    body_name: str = None

    # get pose from mujoco
    def get_pose(self):
        if self.body_name.startswith("touch"):
            site_id = mj.mj_name2id(self.mujoco_model, mj.mjtObj.mjOBJ_SITE, self.body_name)
            # Get the world-space position and orientation (rotation matrix)
            cam_position = self.mujoco_data.site_xpos[site_id].copy()
            cam_orientation = self.mujoco_data.site_xmat[site_id].reshape(3, 3).copy()
            # Convert the rotation matrix to quaternion
            cam_orientation = R.from_matrix(cam_orientation).as_quat(scalar_first=True)
            orientation = cam_orientation
            position = cam_position
        else:
            # Get the position and orientation
            position = self.mujoco_data.xpos[self.obj_id].copy()
            orientation = self.mujoco_data.xmat[self.obj_id].reshape(3, 3).copy()
            orientation = R.from_matrix(orientation).as_quat(scalar_first=True)

        # Convert quaternion to Euler angles (default ZYX convention: yaw, pitch, roll)
        orientation = R.from_quat(orientation,scalar_first=True).as_euler('xyz', degrees=False)
        return position, orientation


class Sensor:
    def __init__(
        self,
        width=120,
        height=160,
        background=None,
        config_path=get_digit_config_path(),
        visualize_gui=True,
        show_depth=True,
        zrange=0.002,
        cid=0,
    ):
        """

        :param width: scalar
        :param height: scalar
        :param background: image
        :param visualize_gui: Bool
        :param show_depth: Bool
        :param config_path:
        :param cid: Int
        """
        self.cid = cid
        self.renderer = Renderer(width, height, background, config_path)

        self.visualize_gui = visualize_gui
        self.show_depth = show_depth
        self.zrange = zrange

        self.cameras = {}
        self.nb_cam = 0
        self.objects = {}
        self.object_poses = {}
        self.normal_forces = {}
        self._static = None

    @property
    def height(self):
        return self.renderer.height

    @property
    def width(self):
        return self.renderer.width

    @property
    def background(self):
        return self.renderer.background
    
    def add_camera_mujoco(self, sensor_name, model, data):
        # Get the site ID using its name
        site_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, sensor_name)
        self.cameras[sensor_name] = Link(site_id, -1, self.cid, data, model, sensor_name)
        self.nb_cam += 1

    def add_object_mujoco(self, body_name, model, data):
        mesh_name = body_name + "_mesh"
        # get object from mujoco
        mesh_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_MESH,mesh_name)
        print(f"mesh_id: {mesh_id}",mesh_name)
        body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, body_name)
        obj_trimesh = self.build_trimesh_from_mujoco(model,mesh_id)
        self.objects[body_name] = Link(body_id, -1, self.cid,data,model, body_name=body_name)
        position, orientation = self.objects[body_name].get_pose()
        # Add object in pyrender
        self.renderer.add_object(
            obj_trimesh,
            body_name,
            position=position,
            orientation=orientation,
        )

    def add_body_mujoco(self, body, model, data):
        self.add_object_mujoco(body, model, data)

    def build_trimesh_from_mujoco(self,model, mesh_id):
        """
        Create a trimesh object from MuJoCo mesh data.

        Parameters:
            model: mjModel
                The MuJoCo model containing mesh data.
            mesh_id: int
                The index of the mesh to extract.

        Returns:
            trimesh.Trimesh: The constructed trimesh object.
        """
        # Get starting index and number of vertices for the mesh
        start_vert = model.mesh_vertadr[mesh_id]
        num_vert = model.mesh_vertnum[mesh_id]
        print(f"start_vert: {start_vert}, num_vert: {num_vert}")
        
        # Extract vertices (reshape to Nx3 array)
        print(model.mesh_vert.shape)
        vertices = model.mesh_vert[start_vert: start_vert + num_vert].reshape(-1, 3)
        print(vertices.shape)
        # switch up x and z axis
        vertices = vertices[:, [2, 1, 0]]
        # Get starting index and number of faces for the mesh
        start_face = model.mesh_faceadr[mesh_id]
        num_face = model.mesh_facenum[mesh_id]
        
        # Extract faces (reshape to Mx3 array of vertex indices)
        faces = model.mesh_face[start_face :start_face + num_face].reshape(-1, 3)

        # Create the trimesh object
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
        #show the mesh in a standalone window
        return mesh
    def loadURDF(self, *args, **kwargs):
        warnings.warn(
            "\33[33mSensor.loadURDF is deprecated. Please use body = "
            "pybulletX.Body(...) and Sensor.add_body(body) instead\33[0m."
        )
        """
        Load the object urdf to pybullet and tacto simulator.
        The tacto simulator will create the same scene in OpenGL for faster rendering
        """
        urdf_fn = args[0]
        globalScaling = kwargs.get("globalScaling", 1.0)

        # Add to pybullet
        obj_id = p.loadURDF(physicsClientId=self.cid, *args, **kwargs)

        # Add to tacto simulator scene
        self.add_object(urdf_fn, obj_id, globalScaling=globalScaling)

        return obj_id

    def update(self):
        warnings.warn(
            "\33[33mSensor.update is deprecated and renamed to ._update_object_poses()"
            ", which will be called automatically in .render()\33[0m"
        )

    def _update_object_poses(self):
        """
        Update the pose of each objects registered in tacto simulator
        """
        for obj_name in self.objects.keys():
            self.object_poses[obj_name] = self.objects[obj_name].get_pose()

    def get_force_mujoco(self, sensor_name, model, data):
        # Fetch touch grid data
        sensor_id = model.sensor(sensor_name).id
        touch_data = data.sensordata[
            sensor_id : sensor_id + model.sensor_dim[sensor_id]
        ].reshape(
            (120, 160, 3)
        ) 
        touch_data = touch_data[:, :, 0] # get only the normal forces
        # get the object names in contact with the sensor
        # contact_names = data.contact_names # TODO: get the contact names from mujoco
        touch_data = {"can": touch_data}
        return touch_data

    @property
    def static(self):
        if self._static is None:
            colors, _ = self.renderer.render(noise=False)
            depths = [np.zeros_like(d0) for d0 in self.renderer.depth0]
            self._static = (colors, depths)

        return self._static

    def _render_static(self):
        colors, depths = self.static
        colors = [self.renderer._add_noise(color) for color in colors]
        return colors, depths

    def render(self, model=None, data=None):
        """
        Render tacto images from each camera's view.
        """

        self._update_object_poses()

        colors = []
        depths = []
        for i in range(self.nb_cam):
            cam_name = f"touch"

            # get the contact normal forces
            normal_forces = self.get_force_mujoco(cam_name, model,data)
            if normal_forces is not None:
                position, orientation = self.cameras[cam_name].get_pose()
                self.renderer.update_camera_pose(position, orientation,cam_name)
                color, depth= self.renderer.render(self.object_poses, normal_forces)
            # Remove the depth from curved gel
                for j in range(len(depth)):
                    depth[j] = self.renderer.depth0[j] - depth[j]
            else:
                color, depth = self._render_static()

            colors += color
            depths += depth

        return colors, depths

    def _depth_to_color(self, depth):
        gray = (np.clip(depth / self.zrange, 0, 1) * 255).astype(np.uint8)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    def updateGUI(self, colors, depths):
        """
        Update images for visualization
        """
        if not self.visualize_gui:
            return

        # concatenate colors horizontally (axis=1)
        color = np.concatenate(colors, axis=1)

        if self.show_depth:
            # concatenate depths horizontally (axis=1)
            depth = np.concatenate(list(map(self._depth_to_color, depths)), axis=1)

            # concatenate the resulting two images vertically (axis=0)
            color_n_depth = np.concatenate([color, depth], axis=0)

            cv2.imshow(
                "color and depth", cv2.cvtColor(color_n_depth, cv2.COLOR_RGB2BGR)
            )
        else:
            cv2.imshow("color", cv2.cvtColor(color, cv2.COLOR_RGB2BGR))

        cv2.waitKey(1)
