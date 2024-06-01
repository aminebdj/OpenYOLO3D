# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import platform
import shutil
import sys
import warnings
from setuptools import find_packages, setup

setup(
    name="mask3d",
    version="0.1",  # Consider using semantic versioning
    packages=find_packages(),
    package_data={"": ["*.yaml"]},
    install_requires=[
        # List your dependencies here, e.g.,
        # 'numpy',
        # 'pandas',
    ],
    include_package_data=True,
    # zip_safe=False,
)
