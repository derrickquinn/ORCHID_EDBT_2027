from setuptools import Extension, setup

from setuptools.command.build_ext import build_ext
from setuptools import find_packages
import os


def get_pybind_include():
    try:
        import pybind11
    except ImportError as exc:
        raise RuntimeError(
            "pybind11 is required to build utils_cpp. "
            "Install it with 'pip install pybind11'."
        ) from exc
    return pybind11.get_include()


class BuildExt(build_ext):
    c_opts = {
        "msvc": ["/O2", "/std:c++17"],
        "unix": ["-O3", "-std=c++17"],
    }
    l_opts = {
        "msvc": [],
        "unix": [],
    }

    def build_extensions(self):
        ct = self.compiler.compiler_type
        opts = self.c_opts.get(ct, [])
        lopts = self.l_opts.get(ct, [])

        if ct == "unix":
            opts.append("-mavx2")
            if os.environ.get("UTILS_CPP_NO_OPENMP") != "1":
                opts.append("-fopenmp")
                lopts.append("-fopenmp")
        for ext in self.extensions:
            ext.extra_compile_args = list(ext.extra_compile_args or []) + opts
            ext.extra_link_args = list(ext.extra_link_args or []) + lopts
        build_ext.build_extensions(self)


ext_modules = [
    Extension(
        "utils_cpp._cluster_legals",
        ["src/cluster_legals_pybind.cpp"],
        depends=[
            "src/cluster_legals_kernels.h",
            "src/predicate_eval_kernels.h",
            "src/topk_kernels.h",
        ],
        include_dirs=[get_pybind_include()],
        language="c++",
    )
]

setup(
    name="utils_cpp",
    version="0.1.0",
    description="C++ kernels for cluster legals",
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExt},
    zip_safe=False,
)
