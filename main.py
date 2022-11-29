from src.Training import TrainingEnv
import numpy as np
from os import path

params = {"memory_size": 400_000,
            "gamma": 0.99,
            "epsilon": [1.0,0.05,30000],
            "batch_size": 64,
            "num_episodes": 60_000,
            "max_sim_time": 6,
            "z_height": 0.5,
            "env_path": path.join(path.abspath(path.curdir),"sim_env","environment.xml"),
            "image_size": 600,
            "cam_R":  np.array([ [0., 0., -1.], [0., 1., 0.],[-1., 0., 0.]]),
            "cam_t": np.array([[0.7],[ 0. ],[4. ]]),
            "fps": 60,
            "floor_collision_threshold": 0.2,
            "target_update_gap": 600,
            "velocity_factor":0.7,
            "render": False, }

def main():
    train_env = TrainingEnv(params=params)
    train_env.train()
        



if __name__ == "__main__":
    main()
        
    

    






 