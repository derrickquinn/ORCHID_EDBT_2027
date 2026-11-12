"""Python packaging wrapper; native compilation is handled by CMake."""

import os
from pathlib import Path
import shlex
import subprocess
import sys

from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext


class BuildExt(build_ext):
    def build_extension(self, ext):
        source_dir = Path(__file__).resolve().parent
        output_dir = Path(self.get_ext_fullpath(ext.name)).resolve().parent
        config = "Debug" if self.debug else "Release"
        build_dir = Path(self.build_temp).resolve() / ext.name
        build_dir.mkdir(parents=True, exist_ok=True)
        subprocess.check_call([
            "cmake", "-S", str(source_dir), "-B", str(build_dir),
            f"-DPython_EXECUTABLE={sys.executable}",
            f"-DCMAKE_BUILD_TYPE={config}",
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={output_dir}",
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{config.upper()}={output_dir}",
            *shlex.split(os.environ.get("CMAKE_ARGS", "")),
        ])
        command = ["cmake", "--build", str(build_dir), "--config", config]
        if self.parallel:
            command.extend(["--parallel", str(self.parallel)])
        subprocess.check_call(command)


setup(
    name="utils_cpp",
    version="0.1.0",
    description="C++ kernels for cluster legals",
    packages=find_packages(),
    ext_modules=[Extension("utils_cpp._cluster_legals", sources=[])],
    cmdclass={"build_ext": BuildExt},
    zip_safe=False,
)
