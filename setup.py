from setuptools import setup, find_packages

setup(
    name="Reinforcement_learning_OpenAIGymnasium",
    version="0.1",
    packages= find_packages(),
    install_requires=[
        "tqdm",
        "swig",
        "numpy==1.26.4",
        "gymnasium[other]",
        "gymnasium[mujoco]",
        "stable-baselines3[extra] == 2.4.0", # Stable baselines 3 library
        "sb3-contrib",
        "pathvalidate"
    ]
)