# SAC_Robot_Exploration
A robot exploration simulator based on SAC

Reference:
https://github.com/Roythuly/off-policy
https://github.com/RobustFieldAutonomyLab/DRL_robot_exploration

How to use:
pip install gradio typing-extensions


To Do:
gpt-feedback class 封装
main函数中基于i_episode取图得到gpt-feedback
replay memory中sample时重新计算reward
gpt-feedback 增加多轮对话功能，根据上下文进行回答