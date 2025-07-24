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
import trimesh

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
    vertices = np.array(
        [
            [0, 0, 0],  # Origin
            [size, 0, 0],  # X-axis
            [0, size, 0],  # Y-axis
            [0, 0, size],  # Z-axis
        ]
    )
    # Define the indices for the lines
    indices = np.array([[0, 1], [0, 2], [0, 3]])  # X-axis  # Y-axis  # Z-axis
    # Define colors for each axis (RGB)
    colors = np.array(
        [
            [255, 0, 0],  # Red for X-axis
            [0, 255, 0],  # Green for Y-axis
            [0, 0, 255],  # Blue for Z-axis
        ]
    )
    # Create a mesh from the lines
    # Create a primitive for each axis
    primitives = []
    for i, (start, end) in enumerate(indices):
        primitives.append(
            pyrender.Primitive(
                positions=vertices[[start, end]],
                color_0=colors[i],
                mode=pyrender.constants.GLTF.LINES,
            )
        )
    # Create a mesh from the primitives
    return pyrender.Mesh(primitives=primitives, is_visible=True)


@dataclass
class Link:
    """
    Dataset class for objects in MuJoCo.
    """
    obj_id: int  # pybullet ID
    link_id: int  # pybullet link ID (-1 means base)
    cid: int  # physicsClientId
    mujoco_data: any = None
    mujoco_model: any = None
    body_name: str = None
    is_camera: bool = False

    # get pose from mujoco
    def get_pose(self):
        """
        Gets the pose of the object in world coordinates, with x-axis flipped for pyrender.
        """
        if self.is_camera:
            # Camera is created from a site, so we need to access a different data
            site_id = mj.mj_name2id(
                self.mujoco_model, mj.mjtObj.mjOBJ_SITE, self.body_name
            )
            # Get the world-space position and orientation (rotation matrix)
            position = self.mujoco_data.site_xpos[site_id].copy()
            orientation = self.mujoco_data.site_xmat[site_id].reshape(3, 3).copy()
            
        else:
            # For bodies, just xpos / xmat is fine
            position = self.mujoco_data.xpos[self.obj_id].copy()
            orientation = self.mujoco_data.xmat[self.obj_id].reshape(3, 3).copy()
        
        # Convert quaternion to Euler angles (default ZYX convention: yaw, pitch, roll)
        orientation = R.from_matrix(orientation).as_quat(scalar_first=True)
        orientation = R.from_quat(orientation, scalar_first=True).as_euler(
            "xyz", degrees=False
        )
        
        # pyrender's x-axis is flipped (LHS), so we need to flip the x-axis on pos and orientation
        position[0] = -position[0]
        orientation[1] = -orientation[1]  # Flip x-axis for MuJoCo
        orientation[2] = -orientation[2]  # Flip x-axis for MuJoCo
        return position, orientation


class Sensor:
    def __init__(
        self,
        width: int = 120,
        height: int = 160,
        background: str = None,
        config_path=get_digit_config_path(),
        visualize_gui: bool = True,
        show_depth: bool = True,
        zrange: float = 0.002,
        cid: int = 0,
    ):
        """
        Initializes the tacto sensor.

        :param width:int
            Width of the sensor image
        :param height:int
            Height of the sensor image
        :param background:image = None
            Background image for the sensor
        :param config_path:str = <default config file>
            Path to the configuration file for the sensor. Uses the default config if not provided.
        :param visualize_gui: bool = True
            Visualizes the rendered sensor images in a GUI window
        :param show_depth:bool = True
            If True, shows the depth image in the GUI window
        :param zrange:float = 0.002
            Depth value used to normalize the depth image
        :param cid:int = 0,
            Client ID for the sensor
        """
        self.cid = cid
        self.renderer = Renderer(width, height, background, config_path)

        self.visualize_gui = visualize_gui
        self.show_depth = show_depth
        self.zrange = zrange

        self.cameras = {}
        self.camera_names = []
        self.tacto_body_ids = (
            {}
        )  # Set of body names, whose sites are used for tacto cameras
        self.object_body_ids = (
            set()
        )  # Set of objects of interest that may come in contact with tacto

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
        """
        Add a camera for pyrender to the sensor. Doesn't actually add a camera to the mujoco model.
        In addition, we store the associated site's body_id in tacto_body_ids for later use.
        :param sensor_name: str
            Name of the sensor to be added. This is defined as a mujoco.sensor.touch_Grid plugin, and its name
              should match the name of its associated site in the mujoco model.
        :param model: mjModel
        :param data: mjData
        """
        # Get the site ID using its name
        site_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, sensor_name)
        # Create the camera to be passed to pyrender
        self.cameras[sensor_name] = Link(
            site_id, -1, self.cid, data, model, sensor_name, True
        )
        # Keep track of the number of cameras
        self.nb_cam = len(self.cameras.keys())
        # Remember what the associated site's body_id is for contact checking
        self.tacto_body_ids[sensor_name] = model.site_bodyid[site_id]
        self.camera_names = list(self.cameras.keys())

    def add_object_mujoco(self, body_name, model, data, mesh_name=None):
        """
        Add an object to the list of objects to be tracked by the sensor.
        The given body_name is used to find the corresponding mesh's name as defined in the xml, by appending _mesh.
        e.g. if body_name is "box_geom", the mesh name must be "box_geom_mesh".
        This mesh is passed to pyrender for rendering the tacto image.
        Since it requires the corresponding object body's pose in the simulation, at least one mj_step should be called
        before this function.

        :param body_name: str
            Name of the body to be added. This is defined as a mujoco body, and its associated mesh is expected to be
            defined in the mujoco model with the name body_name + "_mesh", unless provided otherwise.
        :param model: mjModel
        :param data: mjData
        :param mesh_name: str, optional
            Name of the mesh to be used for the object. If not provided, it defaults to
            body_name + "_mesh". This is useful if the mesh name differs from the default convention
            of appending "_mesh" to the body name.
        """
        mesh_name = body_name + "_mesh" if mesh_name is None else mesh_name
        # get object from mujoco
        mesh_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_MESH, mesh_name)
        print(f"mesh_id: {mesh_id}", mesh_name)
        body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, body_name)
        self.object_body_ids.add(body_id)
        obj_trimesh = self.build_trimesh_from_mujoco(model, mesh_id)
        self.objects[body_name] = Link(
            body_id, -1, self.cid, data, model, body_name=body_name
        )
        position, orientation = self.objects[body_name].get_pose()
        # Add object in pyrender
        self.renderer.add_object(
            obj_trimesh,
            body_name,
            position=position,
            orientation=orientation,
        )

    def add_body_mujoco(self, body, model, data, mesh_name=None):
        self.add_object_mujoco(body, model, data, mesh_name=mesh_name)

    def build_trimesh_from_mujoco(self, model, mesh_id):
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
        vertices = model.mesh_vert[start_vert : start_vert + num_vert].reshape(-1, 3)
        print(vertices.shape)
        # switch up x and z axis
        vertices = vertices[:, [2, 1, 0]]
        # Get starting index and number of faces for the mesh
        start_face = model.mesh_faceadr[mesh_id]
        num_face = model.mesh_facenum[mesh_id]

        # Extract faces (reshape to Mx3 array of vertex indices)
        faces = model.mesh_face[start_face : start_face + num_face].reshape(-1, 3)

        # Create the trimesh object
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
        return mesh

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
        """
        Runs a contact check between the sensor and the objects in the scene.
        If a contact between the sensor and an object of interest is found,
        it fetches the touch grid data from the mujoco sensor and returns it.
        Else, it returns None, to prevent unnecessary rendering of the sensor.

        """
        sensor_body_id = self.tacto_body_ids[sensor_name]
        b1 = None
        b2 = None
        b1_name = None
        b2_name = None
        got_contact = False
        if len(data.contact) == 0:
            return None
        for c in data.contact:
            b1 = model.geom_bodyid[c.geom1]
            b2 = model.geom_bodyid[c.geom2]
            b1_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, b1)
            b2_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, b2)
            if (b1 == sensor_body_id or b1 in self.object_body_ids) and (
                b2 == sensor_body_id or b2 in self.object_body_ids
            ):
                # If the contact is between tacto body and object body, we are interested in the force data
                got_contact = True
        if not got_contact:
            return None

        # Fetch touch grid data
        sensor_id = model.sensor(sensor_name).id
        touch_data = data.sensordata[
            sensor_id : sensor_id + model.sensor_dim[sensor_id]
        ].reshape((120, 160, 3))
        touch_data = touch_data[:, :, 0]  # get only the normal forces

        # get the object names in contact with the sensor
        obj_name = b1_name if b2 == sensor_body_id else b2_name
        touch_data = {obj_name: touch_data}
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
            cam_name = self.camera_names[i]

            # get the contact normal forces
            normal_forces = self.get_force_mujoco(cam_name, model, data)
            if normal_forces is not None:
                position, orientation = self.cameras[cam_name].get_pose()
                # Fixed orientation = object itself rotates around, doesn't fix the translation axis issue
                # orientation = [-3.14100000e+00, -9.43891413e-07, -3.13999998e+00]
                self.renderer.update_camera_pose(position, orientation, cam_name)
                color, depth = self.renderer.render(self.object_poses, normal_forces)
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
