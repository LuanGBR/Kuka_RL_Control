from handlers import *
import random
import numpy as np
import tqdm

def main():
    sim = MujocoHandler("/home/luangb/Documents/TCC/Kuka_RL_Control/environment.xml")
    cam = RGBD_CamHandler(sim,size=600,windowed=False,fps=30)
    cam.R = np.array([ [0., 0., -1.], [0., 1., 0.],[-1., 0., 0.]])
    cam.t = np.array([[0.7],[ 0. ],[4. ]])
    tracker = BallTracker(cam)

    measures = []
    positions = []


    for _ in tqdm.tqdm(range(30_000)):

        sim.reset()
  
        fovx = np.deg2rad(55)
        fovy = np.deg2rad(55)
        theta = random.uniform(-fovx/2,fovx/2)
        r = random.uniform(1,6)
        y = r*np.sin(theta)
        y = np.clip(y,-2.9,1.9)
        x = r*np.cos(theta)
        x = np.clip(x,-1.9,1.9)
        phi = random.uniform(-fovy/2,fovy/2)
        z = r*np.sin(phi)
        z = np.clip(z,0.1,2.15)

        sim.set_state(init_pos=[x, y, z], init_vel=[0, 0, 0])


        cam.update_frame(simulate_fps=False)

        if tracker.update_track():
            pixel = tracker._pixels[-1]
            coords = tracker._positions[-1]
            measures.append(coords)
            positions.append(np.array([x,y,z]))
    np.save("measures.npy",measures)
    np.save("positions.npy",positions)

    print("[x,y,z]")
    cov = np.cov(np.array(measures)-np.array(positions),rowvar=False)
    print("var: ",cov)

if __name__ == "__main__":
    main()
    




