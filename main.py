from src.Training import TrainingEnv
import numpy as np
from os import path

params = {"memory_size": 64000,
            "gamma": 0.99,
            "epsilon": [0.9,0.005,.999],
            "tau": 0.1,
            "batch_size": 128,
            "num_episodes": 20000,
            "max_sim_time": 6,
            "learning_rate": 0.0001,
            "z_height": 0.6,
            "env_path": path.join(path.abspath(path.curdir),"sim_env","environment.xml"),
            "image_size": 600,
            "cam_R":  np.array([ [0., 0., -1.], [0., 1., 0.],[-1., 0., 0.]]),
            "cam_t": np.array([[0.7],[ 0. ],[4. ]]),
            "fps": 60,
            "floor_collision_threshold": 0.2,
            "update_gap": 4,
            "velocity_factor":1,
            "render": False, }

def main():
    train_env = TrainingEnv(params=params)
    train_env.train()
        



if __name__ == "__main__":
    main()
        
    

    






 