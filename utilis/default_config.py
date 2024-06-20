from utilis.config import Config

default_config = Config({
    "seed": 0,
    "tag": "default",
    "start_steps": 5e4,
    "cuda": True,
    "num_steps": 1000001,
    "save": True,
    
    # "env_name": "HalfCheetah-v2", 
    "env_name": "RobEnv",
    "eval": True,
    "eval_episodes": 10,
    "eval_times": 100,
    "replay_size": 200,

    "algo": "SAC",
    "policy": "Gaussian",   # 'Policy Type: Gaussian | Deterministic (default: Gaussian)'
    "gamma": 0.99, 
    "tau": 0.005,
    "lr": 0.00001,
    "alpha": 0.2,
    "automatic_entropy_tuning": True,
    "batch_size": 128, 
    "updates_per_step": 1,
    "target_update_interval": 1,
    "hidden_size": 256
})
