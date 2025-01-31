# Reinforcement_Learning_OpenAI
This repository includes reinforcement learning algorithms and environments that are compatible with OpenAI Gymnasium and the Stable-Baselines3 library.

# Steps to use project
## 0. Installation Requirements

### 0.1 Install python 
Install python using the following link: https://www.python.org/downloads/.

Note: To use the library Stable Baselines3 which is needed for this project, take care of the following points:

    - Stable Baselines3 requires python 3.9+
    - The python version 3.13.1 is not compatible with this library. 
    - The maximum compatible version which i tried was 3.12.0

### 0.2 Install git
Install git using the following link: https://git-scm.com/downloads


### 0.3 (Only for Windows Users) C++ Compiler Support 
To use library the Stable Baseline3 you need a C++ Compiler Support. I used the Build tools from Visual Studio https://visualstudio.microsoft.com/de/downloads/ to install the needed tools but maybe there is another faster option.


### 1. Create a virtual python environment (This is optional but recommended)
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
source <virtual-environment-name>/bin/activate 
```

For Windows users: 
```bash
.\<virtual-environment-name>\Scripts\activate
```

Note: You might need to enable running scripts: 
```bash
set-executionpolicy remotesigned
```

For more details, view https://docs.python.org/3/library/venv.html or https://mothergeo-py.readthedocs.io/en/latest/development/how-to/venv-win.html(Windows)

## 2. Setup project
### 2.1 Clone project:
```bash
git clone https://github.com/DerinHD/Reinforcement_Learning_OpenAI.git 
```
or 
```bash
git clone git@github.com:DerinHD/Reinforcement_Learning_OpenAI.git
```

### 2.2 Install project
Run the following code at the root directory:
```bash
pip install -e .
```

## 3 Project ready
At the core folder, you can run the following files:
- python main.py: Train or evaluate a model on a specific environment via terminal
- python visalization.py: Visualize performance of trained model on a specific environment

