from hatchling.builders.hooks.plugin.interface import BuildHookInterface
from setuptools import Extension
import os
import pybind11


class CustomBuildHook(BuildHookInterface):
    def initialize(self, version, build_data):
        ext_modules = [
            Extension(
                "acd.core._acd",
                sources=[
                    "src/acd/core/bindings.cpp",
                    "src/acd/core/acd.cpp",
                ],
                include_dirs=[
                    pybind11.get_include(),
                    "src/acd/core",
                    os.environ.get("EIGEN_DIR", "third_party/eigen-3.4.1"),
                ],
                language="c++",
                extra_compile_args=["-O3", "-std=c++17"],
            )
        ]

        # Build in-place using setuptools
        from setuptools import setup

        setup(
            script_args=["build_ext", "--inplace"],
            ext_modules=ext_modules,
        )
