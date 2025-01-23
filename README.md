# Reinforcement_Learning_OpenAI
This repository includes reinforcement learning algorithms and environments that are compatible with OpenAI Gymnasium and the Stable-Baselines3 library.

# Steps to use project
## 0. Prerequisites

### 0.1 Install python
https://www.python.org/downloads/

### 0.2 Create a virtual python environment (This is optional but recommended)
1. Install virtualenv:
```bash
pip install virtualenv
```
2. create a new project folder, cd to this folder and run the following code:
```bash
python<version> -m venv <virtual-environment-name> 
```
3. Run the following code to activate the environment:
```bash
source env/bin/activate (can be different for windows)
```

For more details, view https://docs.python.org/3/library/venv.html or https://mothergeo-py.readthedocs.io/en/latest/development/how-to/venv-win.html(Windows)

## 1. Setup project
### 1.1 Clone project:
```bash
git clone https://github.com/DerinHD/Reinforcement_Learning_OpenAI.git 
```
or 
```bash
git clone git@github.com:DerinHD/Reinforcement_Learning_OpenAI.git
```

### 1.2 Install project
Run the following code at the root directory:
```bash
pip install -e .
```

## 3 Project ready
At the core folder, you can run the following files:
- python main.py: Train or evaluate a model on a specific environment via terminal
- python visalization.py: Visualize performance of trained model on a specific environment

