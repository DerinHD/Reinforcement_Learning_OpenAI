# Reinforcement Learning Codebase
This project introduces the RL codebase, a framework that provides user-friendly access to explore and evaluate different RL features 
(e.g., train an agent). The codebase effectively integrates the RL models and environments, so that various combinations can be tested. The implementation is based on the environment interface from the **OpenAI Gymnasium library** and the model interface from **Stable-baselines3**. The framework was implemented such that a customizable training pipeline allows the user to configure the parameter settings of the model and environment easily. Additionally, storing and recording the results of the trained models is possible, so that they can be investigated further for visualization or evaluation purposes. I have tried to focus on the extensibility of the codebase since it should be intuitive for the user to integrate new custom models and environments. The codebase is written in Python. Overall, this codebase aims to automate RL experiments and use them for future research. 

For more information about the **internal structure of the code base** and the **setup of the project**, view the pdf document **RL_codebase_documentation** which is included in the root of the project folder.

**If there are any questions, improvements or ideas for new models, feel free to contact me.**

Note: I have used the GitHub Copilot extension tool from Visual Studio Code as a support to generate and document code.
