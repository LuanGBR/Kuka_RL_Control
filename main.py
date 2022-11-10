#Train robot to catch ball on mujoco with RL

import mujoco_py as mjp
import random
import os

def main():
    # Load model
    model = mjp.load_model_from_path("model.xml")
    sims = [mjp.MjSim(model)]




    

if __name__ == "__main__": 
    main()