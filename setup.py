from setuptools import setup, find_packages
from distutils.extension import Extension
import os
import numpy

import pybind11
pybind11_include = pybind11.get_include()  # 返回 .../site-packages/pybind11/include
eigen_include = '/usr/include/eigen3'   # Ubuntu 20.04 默认路径

# -----------  可选编译 MPC  ------------
BUILD_MPC = os.environ.get("LEGGED_GYM_BUILD_MPC", "0") == "1"
ext_modules = []
if BUILD_MPC:
    ext_modules = [
        Extension(
            "mpc_osqp",
            sources=["legged_gym/MPC_Controller/convex_MPC/mpc_osqp.cc"],
            include_dirs=[
                "legged_gym/MPC_Controller/include",
                "/usr/include/eigen3",
                "extern/osqp/include",
                pybind11.get_include(),
            ],
            language="c++",
            extra_compile_args=["-std=c++17"],
        )
    ]


# -------------------------------------

setup(
    name="legged_gym",
    version="1.0.0",
    author="ETHZ-RSL",
    license="BSD-3-Clause",
    packages=find_packages(),
    author_email="rudinn@ethz.ch",
    description="Isaac Gym environments for Legged Robots",
    install_requires=[
        "osqp",
        "rsl_rl",
    ],
    ext_modules=ext_modules,   # 空列表时不会编译
)