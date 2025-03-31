from setuptools import setup
from glob import glob

setup(
    name="multiple-sclerosis-research",
    version="1.0.0",
    author="Tal Erez",
    author_email="tal.erez@duke.edu",
    description="Multiple Sclerosis Research",
    url="https://github.com/notthattal/multiple-sclerosis-research",
    py_modules=[
        "data_preprocessing",
        "deep_learning_model",
        "naive_model",
        "traditional_model",
        "train_models"
    ],
    package_dir={'': 'scripts'},
    install_requires=[
        "numpy==2.0.2",
        "pandas==2.2.3",
        "scikit-learn==1.6.1",
        "torch==2.6.0",
        "scipy==1.13.1",
    ],
    entry_points={
        "console_scripts": [
            "train-models=train_models:main",
        ],
    },
    python_requires=">=3.9",
    include_package_data=True,
    data_files=[
         ('data/raw_data/MSGaits', glob('data/raw_data/MSGaits/*')),
         ('data/raw_data/NormativeGaits', glob('data/raw_data/NormativeGaits/*')),
         ('data/processed_data', glob('data/processed_data/*'))
    ],
)