# Common Scripts for Prometheus
# github: 
# https://github.com/prometheus-Lab-HKUST-GZ
# reference: 
# https://arxiv.org/abs/2007.04940
# https://github.com/pybind/python_example/blob/master/setup.py
# https://github.com/tohtsky/irspack/blob/main/setup.py
import os
from pathlib import Path

# Available at setup time due to pyproject.toml
from pybind11 import get_cmake_dir
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup, find_packages

__version__ = "1.2.1"
SETUP_DIRECTORY = Path(__file__).resolve().parent

# setup Eigen
# reference: https://github.com/tohtsky/irspack/blob/main/setup.py
class get_eigen_include(object):
    EIGEN3_URL = "https://gitlab.com/libeigen/eigen/-/archive/3.3.7/eigen-3.3.7.zip"
    LIB_DIRNAME = "3rdparty"
    EIGEN3_DIRNAME = "eigen-3.3.7"

    def __str__(self) -> str:
        eigen_include_dir = os.environ.get("EIGEN3_INCLUDE_DIR", None)

        if eigen_include_dir is not None:
            return eigen_include_dir

        target_dir = SETUP_DIRECTORY / self.LIB_DIRNAME / self.EIGEN3_DIRNAME
        if target_dir.exists():
            return str(target_dir)
        os.makedirs(target_dir, exist_ok=True)

        download_target_dir = SETUP_DIRECTORY / "3rdparty/eigen3.zip"
        print('[setup.py] download eigen from: ')
        print('[setup.py]   %s' % self.EIGEN3_URL)
        print('[setup.py] to: %s' % download_target_dir)

        import zipfile
        import requests

        response = requests.get(self.EIGEN3_URL, stream=True)
        with download_target_dir.open("wb") as ofs:
            for chunk in response.iter_content(chunk_size=1024):
                ofs.write(chunk)

        unzip_dir = str(SETUP_DIRECTORY / self.LIB_DIRNAME)
        with zipfile.ZipFile(download_target_dir) as ifs:
            ifs.extractall(path=unzip_dir)

        return str(target_dir)

# triangle walk in c++
ext_modules = [
    Pybind11Extension("simple_phongsurf.triwalk", ["simple_phongsurf/src/triangle_walk_py.cpp"],
        define_macros = [('VERSION_INFO', __version__)],
        include_dirs=[get_eigen_include()],
    ),
]

setup(
    name="simple_phongsurf",
    version=__version__,
    author="Neil Z. Shao",
    author_email="initialneil@gmail.com",
    url="",
    description="Simple Phong Surface for Walking on Triangles",
    long_description="",
    # install_requires=["libigl", "opencv-python", "accelerate"],
    packages=find_packages(),
    ext_modules=ext_modules,
    # extras_require={"test": "pytest"},
    # Currently, build_ext only provides an optional "highest supported C++
    # level" feature, but in the future it may provide more features.
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.0",
)
