from setuptools import setup, find_packages

setup(
    name='hyuod',
    version='1.0.0',
    description='Hybrid Underwater Object Detection',
    author='ShanShuhong',
    packages=find_packages(),
    install_requires=[
        'torch>=1.13.0',
        'torchvision>=0.14.0',
        'numpy>=1.21.0',
        'opencv-python>=4.5.0',
        'pycocotools>=2.0.6',
    ],
    python_requires='>=3.8',
)
