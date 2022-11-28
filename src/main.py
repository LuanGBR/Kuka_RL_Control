from Training import TrainingEnv
import numpy as np
from os import path

params = {"memory_size": 5000,
            "gamma": 0.95,
            "epsilon": [1.0,0.03,200],
            "learning_rate": 0.001,
            "tau": 0.125,
            "batch_size": 32,
            "num_episodes": 4000,
            "max_sim_time": 6,
            "z_height": 0.5,
            "env_path": path.join(path.abspath(path.curdir),"sim_env","environment.xml"),
            "image_size": 600,
            "cam_R":  np.array([ [0., 0., -1.], [0., 1., 0.],[-1., 0., 0.]]),
            "cam_t": np.array([[0.7],[ 0. ],[4. ]]),
            "fps": 30,
            "floor_collision_threshold": 0.3,
            "target_update_gap": 6,
            "velocity_factor":0.7 }

def main():
    train_env = TrainingEnv(params=params)
    train_env.train()
        



if __name__ == "__main__":
    main()
        
    

    






 