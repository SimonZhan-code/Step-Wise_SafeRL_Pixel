#!/usr/bin/env python

import sys

from setuptools import setup

assert sys.version_info.major == 3 and sys.version_info.minor >= 6, \
    "Safety Gym is designed to work with Python 3.6 and greater. " \
    + "Please install it before proceeding."

setup(
    name='safety_gym',
    packages=['safety_gym', 'safety_gym.envs', 'safety_gym.bridges', 'safety_gym.xmls'],
    install_requires=[
        'gym>=0.15.3',
        'joblib~=0.14.0',
        'numpy>=1.17.4',
        'xmltodict~=0.12.0',
    ],
    package_data={'': ['xmls/car.xml',
                                 'xmls/doggo.xml',
                                 'xmls/point.xml']}
)
