"""NEXUS Jetson SDK - Setup configuration."""

from setuptools import find_packages, setup

setup(
    name="nexus-sdk",
    version="0.1.0",
    description="NEXUS distributed intelligence platform for marine robotics",
    author="NEXUS Team",
    python_requires=">=3.11",
    packages=find_packages(),
    install_requires=[
        "pyserial>=3.5",
        "numpy>=1.24",
        "scipy>=1.11",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "ruff>=0.1.0",
            "mypy>=1.0",
        ],
    },
)
