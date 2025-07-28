# TACTO: A Fast, Flexible and Open-source Simulator for High-Resolution Vision-based Tactile Sensors

[![License: MIT](https://img.shields.io/github/license/facebookresearch/tacto)](LICENSE)
[![PyPI](https://img.shields.io/pypi/v/tacto)](https://pypi.org/project/tacto/)
[![CircleCI](https://circleci.com/gh/facebookresearch/tacto.svg?style=shield)](https://circleci.com/gh/facebookresearch/tacto)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
<a href="https://digit.ml/">
<img height="20" src="/website/static/img/digit-logo.svg" alt="DIGIT-logo" />
</a>

<img src="/website/static/img/mujoco-tacto.jpg?raw=true" alt="TACTO Simulator" />

Developed & Tested on Ubuntu 22.04 & MuJoCo 3.2.6

This repository is a MuJoCo port of the [TACTO](https://github.com/facebookresearch/tacto) simulator.
It ports over the core functionalities of TACTO, namely the [sensor.py](tacto/sensor.py), which implements the vast majority of MuJoCo interface logic, and the [renderer.py](tacto/renderer.py), which contains only a minor edit for ensuring that the MuJoCo-pyrender coordinate system is aligned.

The updated code has been thoroughly documented for better clarity on how the overall package functions, which means it can be easily adopted to other environments easily. For more information on the package itself, please refer to the original repository.

NOTE: the simulator is not meant to provide a physically accurate dynamics of the contacts (e.g., deformation, friction), but rather relies on existing physics engines.

**For updates and discussions please join the #TACTO channel at the [www.touch-sensing.org](https://www.touch-sensing.org/) community.**


## Installation

For this port, we recommend cloning the repository and installing the package manually:

```bash
# Optionally, create a venv or conda environment
git clone git@github.com:ayadabdalla/tacto.git
cd ${REPOSITORY_ROOT}
pip install -e .
pip install pyopengl==3.1.9 # ignore warning
```
This will install all dependencies for the package. See the `pyproject.toml` file for more information. 

When installing `pyopengl==3.1.9`, you may get warnings regarding `pyrender` requiring `pyopengl==3.1.0`. `pyrender` works fine even with the newer version of `pyopengl`, and we specifically install the newer version for better compatibility with more recent packages. The warning can be safely ignored without affecting the functionality of the sensor.

## Content
This package contain several components:
1) A renderer to simulate readings from vision-based tactile sensors.
2) An API to simulate vision-based tactile sensors in MuJoCo.
3) Mesh models and configuration files for the [DIGIT](https://digit.ml).

## Simulation Logic
The [sensor.py](tacto/sensor.py) implements the MuJoCo-side interface to the original TACTO code, which renders the sensor readings separately in pyrender.
All dependency on PyBullet has been removed in this version. Unlike the original TACTO package, `urdfpy` is also no longer needed (since the package is no longer maintained and often leads to dependency issues on recent systems). All mesh and pose fetching logic have been replaced with native MuJoCo API.

In essence, `sensor.py` fetches the user-defined MuJoCo `geom` object as well as their poses directly via the MuJoCo API, which is then aligned with pyrender's coordinate system to allow for visually accurate rendering.

While a MuJoCo `body` might be a better anaologous entity of PyBullet's `Link` element, it is often the case that the child `geom` component's mesh has pose offset from the parent `body`, even if the child's pose-related parameters are all set to 0, e.g. `<geom ... pos="0 0 0" euler="0 0 0">`. As such, fetching the `geom` object information directly yields more consistent results than `body`.

Both custom mesh files (`mjGEOM_MESH`) and primitives (`mjGEOM_SPHERE/CAPSULE/ELLIPSOID/CYLINDER/BOX`) are supported.

The pyrender-side "camera" is mounted at the user-defined MuJoCo `site` in the XML file. The force sensor component of pyrender is replaced with MuJoCo's `touch-sensor` plugin, which should also be mounted on the same `site` as the camera.

## Code Example
A thoroughly explained example can be found in [examples/demo_mujoco_digit.py](examples/demo_mujoco_digit.py), which contains a simple MuJoCo environment with several example meshes to show off the sensor's functionalities, as well as how the sensor can be integrated into your existing MuJoCO environment.

## Upcoming features
- Multi-sensor examples, including a robotic hand

### Headless Rendering
NOTE: the renderer requires a screen. For rendering headless, use the "EGL" mode with GPU and CUDA driver or "OSMESA" with CPU. 
See [PyRender](https://pyrender.readthedocs.io/en/latest/install/index.html) for more details.

With `pyopengl==3.1.9`, you can directly use headless rendering mode without requiring a patched version like the original Tacto.

You can specify which engine to use for headless rendering (either `'egl'` or `'osmesa'`) like:

```
import os
os.environ["PYOPENGL_PLATFORM"] = "osmesa" # osmesa cpu rendering
# os.environ["PYOPENGL_PLATFORM"] = "egl" # egl gpu rendering

```

## License
This project is licensed under MIT license, as found in the [LICENSE](LICENSE) file.


## Citing
If you use this project in your research, please cite:

```BibTeX
@Article{Wang2022TACTO,
  author   = {Wang, Shaoxiong and Lambeta, Mike and Chou, Po-Wei and Calandra, Roberto},
  title    = {{TACTO}: A Fast, Flexible, and Open-source Simulator for High-resolution Vision-based Tactile Sensors},
  journal  = {IEEE Robotics and Automation Letters (RA-L)},
  year     = {2022},
  volume   = {7},
  number   = {2},
  pages    = {3930--3937},
  issn     = {2377-3766},
  doi      = {10.1109/LRA.2022.3146945},
  url      = {https://arxiv.org/abs/2012.08456},
}
```

