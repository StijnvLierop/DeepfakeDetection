import re
from setuptools import setup, find_packages


if __name__ == '__main__':
    with open("README.md", "r") as f:
        readme = f.read()

    with open("requirements.txt", "r") as f:
        dependencies = list(
            filter(
                lambda x: not re.search(r"^(--|#)", x),
                map(str.strip, f.read().split("\n"))
            )
        )
    setup(
        name="deepfake_detection",
        version="0.0.1",
        description="A library for deepfake detection research.",
        long_description=readme,
        author="Stijn van Lierop",
        author_email="s.van.lierop@nfi.nl",
        url="https://github.com/StijnvLierop/DeepfakeDetection",
        packages=find_packages(exclude=["tests", "tests.*"]),
        install_requires=dependencies,
        extras_require={
            "dev": [
                "pytest",
                "flake8",
            ],
        },
        python_requires=">=3.10",
        classifiers=[
            "Programming Language :: Python :: 3",
        ],
    )
