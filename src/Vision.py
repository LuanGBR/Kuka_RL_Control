import mujoco
import numpy as np
import cv2
import mujoco_viewer
from collections import deque
from filterpy import kalman
from imutils import grab_contours


class RGBD_CamHandler:
    """A handler for the RGBD camera. It is used to get the RGBD images from the environment and the 3D coordinates of a pixel."""
    def __init__(self,environment,size=500,windowed=False,fps=60):
        self._env = environment
        self._env._cam_aux_count = 0
        model = self._env.model
        data = self._env.data
        self._size = size
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
    def size(self):
        return self._size


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
        if self._env._cam_aux_count > self._ticks_per_frame or not simulate_fps:
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
        p = np.array(((point[0],point[1],point[2],1)))
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

    
    def reset(self):
        model = self._env.model
        data = self._env.data
        self._env._cam_aux_count = self.fps + 1
        if self._windowed:
            pass
        else:
            self.img_buffer,self.depth_buffer = self._viewer.read_pixels(camid=0,depth=True)



class BallLostException(Exception):
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
        self._Kalman = kalman.KalmanFilter(dim_x=9, dim_z=3)
        dt = 1/self._cam.fps
        half_dt_squared = 0.5 * dt * dt
        self._Kalman.F = np.array([[1, 0, 0, dt, 0, 0, half_dt_squared, 0, 0],
                                                 [0, 1, 0, 0, dt, 0, 0, half_dt_squared, 0],
                                                 [0, 0, 1, 0, 0, dt, 0, 0, half_dt_squared],
                                                 [0, 0, 0, 1, 0, 0, dt, 0, 0],
                                                 [0, 0, 0, 0, 1, 0, 0, dt, 0],
                                                 [0, 0, 0, 0, 0, 1, 0, 0, dt],
                                                 [0, 0, 0, 0, 0, 0, 1, 0, 0],
                                                 [0, 0, 0, 0, 0, 0, 0, 1, 0],
                                                 [0, 0, 0, 0, 0, 0, 0, 0, 1]], dtype=np.float32)
        self._Kalman.H = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0],
                                                   [0, 1, 0, 0, 0, 0, 0, 0, 0],
                                                   [0, 0, 1, 0, 0, 0, 0, 0, 0]], dtype=np.float32)
        Tau = np.array([half_dt_squared, half_dt_squared, half_dt_squared, dt, dt, dt, 1, 1, 1], dtype=np.float32).reshape((9,1))
        var = 1
        Q = Tau * var* Tau.T
        self._Kalman.Q = Q.astype(np.float32)
        self. _Kalman.P *= 1000

        

        self._Kalman.R = np.array([[ 3.04575243e-03, -8.60463925e-04,  1.25414465e-04],
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
        cnts = grab_contours(cnts)
        center = None
        radius = None

        # only proceed if at least one contour was found
        if len(cnts) > 0:
            
            c = max(cnts, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            radius = int(radius)
            if radius<2:
                return None,None

        
        return center,radius


    

    def update_track(self):
        
        frame = self._cam.get_image_array()
        center,radius = self._find_circles(frame)
        MISSING_THRESHOLD = 40
        if center is None and self._not_found >= MISSING_THRESHOLD:
            self._not_found += 1
            self._pixels.clear()
            self._radius.clear()
            self._positions.clear()
            self._ticks.clear()
            self._i = 0
            raise BallLostException
        elif center is None and self._not_found < MISSING_THRESHOLD:
            self._not_found += 1
            self._Kalman.predict()
            position = np.array(self._Kalman.x[:3]).flatten()
            self._positions.append(position)
            center = self._cam.get_pixel_coords(position).flatten()
            border_position = np.array(position)+np.array([0,0,self._ball_radius])
            radius = center[1] - self._cam.get_pixel_coords(border_position)[1]
        elif center is not None and self._i == 0:
            self._positions.append(self._cam.get_3D_coords(center))
            self._Kalman.x = np.array([self._positions[-1][0],self._positions[-1][1],self._positions[-1][2],0,0,0,0,0,0], dtype=np.float32)
            self._not_found = 0
        elif center is not None and self._i > 0:
            self._Kalman.predict() 
            self._positions.append(self._cam.get_3D_coords(center))
            self._Kalman.update(np.array([self._positions[-1][0],self._positions[-1][1],self._positions[-1][2]], dtype=np.float32))
        self._pixels.append(center)
        self._radius.append(radius)
        self._ticks.append(self._i)
        self._positions.append
        self._i += 1


    def is_tracking(self):
        return len(self._pixels) > 0
    def _is_free_fall(self):
        # fps = self._cam.fps
        # n=4
        # if len(self._positions) <= n:
        #     return False
        
        # if len(self._positions) > n:
        #     zpos = np.array(self._positions)[-(n+1):,2]
        #     vel = np.diff(zpos) * fps
        #     acc = np.diff(vel) * fps
        #     acc = acc.mean()
        #     if np.abs(acc+9.8) < 3 and acc<0:
        #         return True
        #     else:
        #         return False
        if np.abs(self._Kalman.x[8]+9.8) < 2:
            return True 
        else:
            return False
    def ball_position(self):
        if len(self._positions) > 0:
            return self._positions[-1]
        else:
            return None
    
    def ball_velocity(self):
        return np.array(self._Kalman.x[3:6]).flatten()
    
    def reset(self):
        self.__init__(self._cam)
