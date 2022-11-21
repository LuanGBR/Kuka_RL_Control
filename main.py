from handlers import *
import matplotlib.pyplot as plt

def main():
    sim = MujocoHandler("/home/luangb/Documents/TCC/Kuka_RL_Control/environment.xml")
    cam = RGBD_CamHandler(sim,size=600,windowed=False,fps=30)
    cam.R = np.array([ [0., 0., -1.], [0., 1., 0.],[-1., 0., 0.]])
    cam.t = np.array([[0.7],[ 0. ],[4. ]])
   



    while True:
        sim.reset()
        sim.set_random_state()
        tracker = BallTracker(cam)
        while sim.time < 3:
            sim.step()
            if cam.update_frame():
                tracker.update_track()
                tracker.cv2_show(True, True,True)
        



if __name__ == "__main__":
    main()
        
    

    






 