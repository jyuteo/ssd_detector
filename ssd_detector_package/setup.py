from setuptools import setup
import os

main_dir = os.path.abspath(os.path.dirname(__file__))
requirements = open('requirements.txt', 'r')

requirements_ls = [package.strip() for package in requirements.readlines()]

setup(
    name='ssd_detector',
    packages=['ssd_detector'],
    install_requires=requirements_ls,
    classifiers=[
        'Programming Language :: Python :: 3'
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        ],
    python_requires='>=3.6',
    version='1.0.0'
)
