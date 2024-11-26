from setuptools import setup, find_packages

def parse_requirements(filename):
    """Load requirements from a pip requirements file."""
    with open(filename, "r") as f:
        return f.read().splitlines()

setup(
    name="dataset-project",
    version="1.0.0",
    author="Tal Erez",
    author_email="tal.erez@duke.edu",
    description="Dataset Project for AIPI 510s",
    url="https://github.com/notthattal/multiple-sclerosis-research",
    packages=find_packages(),
    install_requires=parse_requirements("requirements.txt"),
    py_modules=["eda"],
    entry_points={
        "console_scripts": [
            "run-eda=eda:main",
        ],
    },
    python_requires=">=3.9"
)