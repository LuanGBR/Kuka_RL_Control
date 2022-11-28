import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import use
use("agg")
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class Utils:
    def plot_3D_points(points):
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(111, projection='3d')
        ax.view_init(10,-30)
        x,y,z = points[:,0],points[:,1],points[:,2]
        ax.scatter(x,y,z, c=x, s=0.1, cmap="inferno", linewidth=0.5)
        ax.set_box_aspect((np.ptp(x), np.ptp(y), np.ptp(z)))
        # ax.set_xlim(-2.1,2.1)
        # ax.set_ylim(-3.1,3.1)
        # ax.set_zlim(0,3.1)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f"Nuvem de pontos")
        return fig,ax

class CV2renderer:
    def __init__(self, enable_bit={"kalman_pred":True, "contour":True, "path":True, "plot":True}, cam=None, tracker=None):
        self._kalman_pred = enable_bit["kalman_pred"]
        self._contour = enable_bit["contour"]
        self._path = enable_bit["path"]
        self._plot = enable_bit["plot"]
        self._cam = cam
        self._tracker = tracker
        self._figsize = (self._cam.size//100, self._cam.size//100)
        self._dpi = 100

    def render(self,durations = None, terminal_rewards = None):
        frame = self._cam.get_image_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        pixels = self._tracker._pixels
        radius = self._tracker._radius
        kalman_x = self._tracker._Kalman.x
        falling = self._tracker._is_free_fall()


        if pixels:
            center,radius = pixels[-1],radius[-1]
            if self._kalman_pred and falling:
                p = kalman_x[:3]
                v = kalman_x[3:6]
                a = kalman_x[6:]
                t = np.linspace(0,1,20)
                pred = [x for x in (p+v*t+0.5*a*t**2 for t in t) if x[2]>0 and x[1]<2]
                pred = [self._cam.get_pixel_coords(p) for p in pred]
                for i in range(1, len(pred)):
                    thickness = 2
                    cv2.line(frame, pred[i - 1], pred[i], (100,100,100), thickness)
            if self._path:
                for i in range(1, len(pixels)):
                    thickness = int(np.sqrt(20 / float((len(pixels))-i + 1)) * 2)
                    cv2.line(frame, pixels[i - 1], pixels[i], (255, 0, 0), thickness)
            if self._contour:
                cv2.circle(frame, center,radius , (0, 255, 0) if  falling else (0,0,255), 2)
            if self._plot:

                #plt clear
                fig, axs = plt.subplots(2, 1, figsize=self._figsize,dpi=self._dpi,sharex=True) 
                durations_t = torch.tensor(durations, dtype=torch.float)
                
                ax,ax2 = axs

                ax.set_title('Training...')
                ax.grid(True)
                ax.set_ylabel('Duration')

                ax2.set_xlabel('Episode')
                ax2.set_ylabel('Last reward')
                ax2.grid(True)

                plt.text(0.05, 0.01, f"pos: ({kalman_x[0]:2.2f},{kalman_x[1]:2.2f},{kalman_x[2]:2.2f}) acc_z: {kalman_x[-1]:1.2f}", fontsize=12, color='red',transform=fig.transFigure)

                if terminal_rewards:
                    terminal_rewards_t = torch.tensor(terminal_rewards, dtype=torch.float)
                    ax2.plot(terminal_rewards_t.numpy())
                
                # Take 100 episode averages and plot them too
                if len(durations_t) >= 100:
                    means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
                    means = torch.cat((torch.zeros(99), means))
                    ax.plot(means.numpy())
                else:
                    ax.plot(durations_t.numpy())

                plt.pause(0.001)  # pause a bit so that plots are updated

                #plot img as np array
                fig.canvas.draw()
                plot = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
                plot = plot.reshape(fig.canvas.get_width_height()[::] + (3,))
                plot = cv2.cvtColor(plot, cv2.COLOR_RGB2BGR)
                frame = np.concatenate((frame,plot),axis=1)
                plt.close(fig)
            cv2.imshow('frame',frame)
            cv2.waitKey(1)

                
                            
        

    
    
    