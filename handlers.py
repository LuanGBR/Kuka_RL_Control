import mujoco
import mujoco_viewer
import numpy as np
import cv2
from cv2 import KalmanFilter
import random
import matplotlib.pyplot as plt
from collections import deque
import imutils

class MujocoHandler:
    """A handler for the mujoco environment."""
    def __init__(self,model_path):
        model = mujoco.MjModel.from_xml_path(model_path)
        data = mujoco.MjData(model)
        self._model = model
        self._data = data
        self._cam_aux_count = 0

    @property
    def model(self):
        return self._model
    
    @property
    def data(self):
        return self._data
    
    @property
    def time(self):
        return self._data.time
    
    @property
    def timestep(self):
        return self._model.opt.timestep
    
    @property
    def ticks(self):
        return self._data.solver_iter
    
    def step(self):
        self._cam_aux_count += 1
        mujoco.mj_step(self._model, self._data)
    
    def set_state(self, init_pos, init_vel):
        self.reset()
        self._data.qpos[0:3] = init_pos
        self._data.qvel[0:3] = init_vel
        mujoco.mj_forward(self._model, self._data)
    
    def reset(self):
        mujoco.mj_resetData(self._model, self._data)
        mujoco.mj_forward(self._model, self._data)
    
    def set_random_state(self):
        init_pos = np.array([0.5,-2,1]) 
        V_abs = np.random.uniform(4,5.5)
        V_theta = np.random.uniform(np.pi/4,np.pi/3)
        vx = 0
        vy = V_abs*np.cos(V_theta)
        vz = V_abs*np.sin(V_theta)
        init_vel = np.array([vx,vy,vz])

        

        self.set_state(init_pos, init_vel)



class RGBD_CamHandler:
    """A handler for the RGBD camera. It is used to get the RGBD images from the environment and the 3D coordinates of a pixel."""
    def __init__(self,environment,size=500,windowed=False,fps=60):
        self._env = environment
        model = self._env.model
        data = self._env.data
        self._width = size
        self._height = size
        self._windowed = windowed
        self._fps = fps
        self._ticks_per_frame = int(1/(self._fps*self._env.timestep))
        mujoco.mj_forward(model,data)
        if self._windowed:
            self._viewer = mujoco_viewer.MujocoViewer(model, data,'window',width=self._width,height=self._height,hide_menus=True)
        else:
            self._viewer = mujoco_viewer.MujocoViewer(model, data,"offscreen",width=self._width,height=self._height)
            self.img_buffer,self.depth_buffer = self._viewer.read_pixels(camid=0,depth=True)
        
        
        f = 0.5 * self._height / np.tan(model.camera("depth_camera0").fovy[0] * np.pi / 360)
        self._K = np.array(((f, 0, self._width / 2), (0, f, self._height / 2), (0, 0, 1)), dtype=np.float32)
        self._R = np.eye(3)
        self._t = np.zeros(3).reshape((3,1))
        self._H = self._calc_H()
        self._Hinv = self._calc_Hinv()
    
    @property
    def fps(self):
        return self._fps


    @property
    def R(self):
        return self._R
    
    @R.setter
    def R(self, R):
        self._R = R
        self._H = self._calc_H()
        self._Hinv = self._calc_Hinv()
    
    @property
    def K(self):
        return self._K
    
    @K.setter
    def K(self, K):
        self._K = K
        self._H = self._calc_H()
        self._Hinv = self._calc_Hinv()

    @property
    def t(self):
        return self._t
    
    @t.setter
    def t(self, t):
        self._t = t.reshape((3,1))
        self._H = self._calc_H()
        self._Hinv = self._calc_Hinv()

    @property
    def H(self):
        return self._H

    @property
    def Hinv(self):
        return self._Hinv
    

    def _calc_H(self):
        Rt = np.concatenate((self._R, self._t), axis=1)
        K_fr = np.eye(4, dtype=np.float32)
        K_fr[:3,:3] = self._K
        Rt_fr = np.eye(4, dtype=np.float32)
        Rt_fr[:3,:4] = Rt
        return K_fr @ Rt_fr
    
    def _calc_Hinv(self):
        return np.linalg.inv(self._H)

    def update_frame(self,simulate_fps=True):
        if  self._env._cam_aux_count > self._ticks_per_frame or not simulate_fps:
            self._env._cam_aux_count = 0
            if self._windowed:
                self._viewer.render()
            else:
                self.img_buffer,self.depth_buffer = self._viewer.read_pixels(camid=0,depth=True)
            return True
        else:
            return False

    def get_image_array(self):
        return self.img_buffer
    
    def get_depth_array(self):
        return self.depth_buffer


    def get_3D_coords(self,pixel):
        """Converts a pixel coordinate to a 3D point in meters."""
        depth = self.real_depth(pixel)
        p= np.array([[pixel[1]],[pixel[0]],[1.0],[1/depth]])
        p =  self.Hinv @ p
        return depth * p[:3].T[0]
    
    def get_pixel_coords(self,point):
        """Converts a 3D point in meters to a pixel coordinate."""
        p = np.array((point[0],point[1],point[2],1))
        p = self.H @ p.T 
        p = p/p[2]
        return np.array([p[1],p[0]],dtype=np.int32)


    def real_depth(self,pixel):
        """Converts a pixel coordinate to a real depth in meters."""
        model = self._env.model
        extent = model.stat.extent
        near = model.vis.map.znear  * extent
        far = model.vis.map.zfar * extent
        try:
            depth_meters = near / (1 - self.depth_buffer[(pixel[1],pixel[0])] * (1 - near / far))
        except IndexError:
            x = np.clip(pixel[1], 0, self._width - 1)
            y = np.clip(pixel[0], 0, self._height - 1)
            depth_meters = near / (1 - self.depth_buffer[x,y] * (1 - near / far))
        return depth_meters[0]
        

    def save_image(self,filename):
        img = self.get_image_array()
        cv2.imwrite(filename,img)
    
    def get_point_cloud(self):
        model = self._env.model
        extent = model.stat.extent
        near = model.vis.map.znear  * extent
        far = model.vis.map.zfar * extent
        self.update_frame()
        depth_meters = near / (1 - self.depth_buffer * (1 - near / far))
        p = np.ones((self._width,self._height,4))
        p[:,:,0] = np.indices((self._width,self._height))[0]
        p[:,:,1] = np.indices((self._width,self._height))[1]
        p[:,:,3] = 1.0/depth_meters[:,:,0]
        p = p.reshape((self._width*self._height,4))
        p = depth_meters.reshape((1,self._width*self._height)) * (self.Hinv @ p.T)
        return p[:3].T

    
    def try_ended(self):
        return self._env.time > 10

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
        


class RLearnHandler:
    def __init__(self,environment):
        self._environment = environment
        

    def get_action(self, state):
        pass

    def get_reward(self, action):
        pass

    def get_state(self):
        pass

    def get_next_state(self, action):
        pass

class BallTracker:
    """Class to track a ball in a video stream. It uses a HSV color space to filter the ball"""
    def __init__(self,cam,ball_radius = 0.0343):
        self._ball_radius = ball_radius
        self._cam = cam 
        self._color_lower = np.array([31, 25.0, 25.0])
        self._color_upper = np.array([41, 254.0, 254.0])
        self._pixels = deque(maxlen=20)
        self._radius = deque(maxlen=20)
        self._positions = deque(maxlen=20)
        self._ticks = deque(maxlen=20)
        self._Kalman = cv2.KalmanFilter(dynamParams=9, measureParams=3, controlParams=0, type=cv2.CV_32F)
        dt = 1/self._cam.fps
        half_dt_squared = 0.5 * dt * dt
        self._Kalman.transitionMatrix = np.array([[1, 0, 0, dt, 0, 0, half_dt_squared, 0, 0],
                                                 [0, 1, 0, 0, dt, 0, 0, half_dt_squared, 0],
                                                 [0, 0, 1, 0, 0, dt, 0, 0, half_dt_squared],
                                                 [0, 0, 0, 1, 0, 0, dt, 0, 0],
                                                 [0, 0, 0, 0, 1, 0, 0, dt, 0],
                                                 [0, 0, 0, 0, 0, 1, 0, 0, dt],
                                                 [0, 0, 0, 0, 0, 0, 1, 0, 0],
                                                 [0, 0, 0, 0, 0, 0, 0, 1, 0],
                                                 [0, 0, 0, 0, 0, 0, 0, 0, 1]], dtype=np.float32)
        self._Kalman.measurementMatrix = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0],
                                                   [0, 1, 0, 0, 0, 0, 0, 0, 0],
                                                   [0, 0, 1, 0, 0, 0, 0, 0, 0]], dtype=np.float32)
        Tau = np.array([half_dt_squared, half_dt_squared, half_dt_squared, dt, dt, dt, 1, 1, 1], dtype=np.float32).reshape((9,1))
        var = 0.1
        Q = Tau * var* Tau.T
        self._Kalman.processNoiseCov = Q.astype(np.float32)

        

        self._Kalman.measurementNoiseCov = np.array([[ 3.04575243e-03, -8.60463925e-04,  1.25414465e-04],
                                                    [-8.60463925e-04,  5.48753112e-04, -3.10123593e-05],
                                                    [ 1.25414465e-04, -3.10123593e-05,  1.01623189e-04]], dtype=np.float32)

        self._i = 0 
        self._not_found = 10
        # Kkalman init pas


                                                   
         
    
    def _find_circles(self, frame):
        pts = []
        blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_RGB2HSV)

        
        mask = cv2.inRange(hsv, self._color_lower, self._color_upper)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        center = None
        radius = None

        # only proceed if at least one contour was found
        if len(cnts) > 0:
            
            c = max(cnts, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            radius = int(radius)

        
        return center,radius


    

    def update_track(self):
        
        frame = self._cam.get_image_array()
        center,radius = self._find_circles(frame)

        if center is None and self._not_found >= 10: 
            self._not_found += 1
            self._pixels.clear()
            self._radius.clear()
            self._positions.clear()
            self._ticks.clear()
            self._i = 0
            return 
        elif center is None and self._not_found < 10:
            self._not_found += 1
            predict = self._Kalman.predict()[0:3]
            position = predict[0:3].flatten()
            self._positions.append(position)
            center = self._cam.get_pixel_coords(position)
            border_position = position+np.array([0,0,self._ball_radius])
            radius = center[1] - self._cam.get_pixel_coords(border_position)[1]
        elif center is not None and self._i == 0:
            self._positions.append(self._cam.get_3D_coords(center))
            self._Kalman.statePost = np.array([self._positions[-1][0],self._positions[-1][1],self._positions[-1][2],0,0,0,0,0,0], dtype=np.float32)
            self._not_found = 0
        elif center is not None and self._i > 0:
            self._Kalman.predict() 
            self._positions.append(self._cam.get_3D_coords(center))
            self._Kalman.correct(np.array([self._positions[-1][0],self._positions[-1][1],self._positions[-1][2]], dtype=np.float32))
        self._pixels.append(center)
        self._radius.append(radius)
        self._ticks.append(self._i)
        self._positions.append
        self._i += 1



    def _is_free_fall(self):
        fps = self._cam.fps
        n=4
        if len(self._positions) <= n:
            return False
        
        if len(self._positions) > n:
            zpos = np.array(self._positions)[-(n+1):,2]
            vel = np.diff(zpos) * fps
            acc = np.diff(vel) * fps
            acc = acc.mean()
            if np.abs(acc+9.8) < 3 and acc<0:
                return True
            else:
                return False
    
    def cv2_show(self,contour=False,path=False):
        frame = self._cam.get_image_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if self._pixels:
            center,radius = self._pixels[-1],self._radius[-1]
            if path:
                for i in range(1, len(self._pixels)):
                    thickness = int(np.sqrt(20 / float((len(self._pixels))-i + 1)) * 2)
                    cv2.line(frame, self._pixels[i - 1], self._pixels[i], (255, 0, 0), thickness)
            if contour:
                cv2.circle(frame, center,radius , (0, 255, 0) if self._is_free_fall() else (0,0,255), 2)
            
        cv2.imshow("frame",frame)
        cv2.waitKey(1)
            
    