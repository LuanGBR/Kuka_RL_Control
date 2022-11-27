import mujoco
import mujoco_viewer
import numpy as np
import cv2
from cv2 import KalmanFilter
import random
import matplotlib.pyplot as plt
from collections import deque
import imutils
from filterpy import kalman
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from functools import lru_cache
from scipy.spatial.transform import Rotation


from keras.callbacks import TensorBoard

#...

# Own Tensorboard class
class ModifiedTensorBoard(TensorBoard):

    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.FileWriter(self.log_dir)

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        pass

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        self._write_logs(stats, self.step)

class MujocoHandler:
    """A handler for the mujoco environment."""
    def __init__(self,model_path):
        model = mujoco.MjModel.from_xml_path(model_path)
        data = mujoco.MjData(model)
        self._model = model
        self._data = data
        self._cam_aux_count = 0
        self._max_velocities = np.array([2.72271363311,2.72271363311,5.75958653158,5.75958653158,5.75958653158,10.7337748998]).astype(np.float32)

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
        self._model.qpos0[0:3] = init_pos
        self.reset()
        self._data.qvel[0:3] = init_vel
        mujoco.mj_forward(self._model, self._data)
    
    def reset(self):
        mujoco.mj_resetData(self._model, self._data)
        mujoco.mj_forward(self._model, self._data)
    
    def set_random_state(self):
        init_pos = np.array([-0.966,-2,1]) 
        V_abs = np.random.uniform(4.5,5.5)
        # V_theta = np.random.uniform(np.pi/4,np.pi/3) # TODO: check if this is the right range
        V_theta = np.pi/4
        vx = 0
        vy = V_abs*np.cos(V_theta)
        vz = V_abs*np.sin(V_theta)
        init_vel = np.array([vx,vy,vz])

        self.set_state(init_pos, init_vel)
        return np.concatenate((init_pos, init_vel))

    def get_ball_state(self):
        return np.concatenate((self._data.body("ball").xpos, self._data.body("ball").cvel[3:6]))
    
    def get_arm_state(self):
        return (self._data.qpos[7:13])

    def get_ball(self):
        return self._env.data.body("ball")

    def take_action(self,action,k):
        ternary = np.base_repr(action, base=3)
        ternary = ternary.zfill(6)
        for i,n in enumerate(ternary):
            self._data.ctrl[i] = (int(n) - 1) * k * self._max_velocities[i]
    
    def get_n_contacts(self):
        return self._data.ncon
    
    def get_body_contacts(self):
        body_pairs =  [(self._model.body(self._model.geom(contact.geom1).bodyid[0]).name,self._model.body(self._model.geom(contact.geom2).bodyid[0]).name) for contact in self._data.contact]
        return body_pairs
    def is_ball_in_target(self):
        body_pairs = self.get_body_contacts()
        if ("ball","target") in body_pairs or ("target","ball") in body_pairs:
            return True
        else:
            return False
    
    def is_ball_on_floor(self):
        body_pairs = self.get_body_contacts()
        if ("ball","world") in body_pairs or ("world","ball") in body_pairs:
            return True
        else:
            return False
    
    def get_basket_position(self):
        return self._data.body("target").xpos
    
    def get_basket_orientation(self):
        return self._data.body("target").xmat.reshape((3,3))
    
    



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
            if radius<2:
                return None,None

        
        return center,radius


    

    def update_track(self):
        
        frame = self._cam.get_image_array()
        center,radius = self._find_circles(frame)
        MISSING_THRESHOLD = 30
        if center is None and self._not_found >= MISSING_THRESHOLD:
            self._not_found += 1
            self._pixels.clear()
            self._radius.clear()
            self._positions.clear()
            self._ticks.clear()
            self._i = 0
            return 
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
    
    def cv2_show(self,contour=False,path=False,kalman_pred=False):
        frame = self._cam.get_image_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if self._pixels:
            falling = self._is_free_fall()
            center,radius = self._pixels[-1],self._radius[-1]
            if kalman_pred and falling:
                p = self._Kalman.x[:3]
                v = self._Kalman.x[3:6]
                a = self._Kalman.x[6:]
                t = np.linspace(0,1,20)
                pred = [x for x in (p+v*t+0.5*a*t**2 for t in t) if x[2]>0 and x[1]<2]
                pred = [self._cam.get_pixel_coords(p) for p in pred]
                for i in range(1, len(pred)):
                    thickness = 2
                    cv2.line(frame, pred[i - 1], pred[i], (100,100,100), thickness)
            if path:
                for i in range(1, len(self._pixels)):
                    thickness = int(np.sqrt(20 / float((len(self._pixels))-i + 1)) * 2)
                    cv2.line(frame, self._pixels[i - 1], self._pixels[i], (255, 0, 0), thickness)
            if contour:
                cv2.circle(frame, center,radius , (0, 255, 0) if  falling else (0,0,255), 2)
            
        cv2.imshow("frame",frame)
        cv2.waitKey(1)
    def reset(self):
        self.__init__(self._cam)


class DQN:
    def __init__(self, params: dict = {}):
        """params format: 
                {memory_size: int,
                gamma: float,
                epsilon: float,
                epsilon_decay: float,
                epsilon_min: float,
                learning_rate: float,
                tau: float,
                batch_size: int,
                observation_space: int,
                action_space: int}"""
        self.observation_space = params.get("observation_space", 6)
        self.action_space = params.get("action_space", 27)
        self._memory = deque(maxlen=params.get('memory_size', 2000))
        self._gamma = params.get('gamma', 0.95)
        self._epsilon = params.get('epsilon', 0.9)
        self._epsilon_decay = params.get('epsilon_decay', 0.98)
        self._epsilon_min = params.get('epsilon_min', 0.01)
        self._learning_rate = params.get('learning_rate', 0.001)
        self._tau = params.get('tau', 0.125)
        self._batch_size = params.get('batch_size', 32)
        self._model = self._create_model()
        self._target_model = self._create_model()
        self._target_train_counter = 0
        self._target_train_interval = 6


    def _create_model(self):
        model   = Sequential()
        state_shape  = 12 # number of state variables
        model.add(Dense(24, input_dim=state_shape, activation="relu"))
        model.add(Dense(48, activation="relu"))
        model.add(Dense(96, activation="relu"))
        model.add(Dense(729)) #number of actions
        model.compile(loss="mean_squared_error",
            optimizer=Adam(lr=self._learning_rate))

        model.summary()
        return model

    def remember(self, state, action, reward, next_state, done):
        self._memory.append((state, action, reward, next_state, done))

    # def replay(self):
    #     if len(self._memory) < self._batch_size:
    #         return
    #     minibatch = random.sample(self._memory, self._batch_size)
    #     for state, action, reward, next_state, done in minibatch:
    #         target = reward
    #         if not done:
    #             target = (reward + self._gamma * 
    #                 np.amax(self._target_model.predict(next_state)[0]))
    #         target_f = self._model.predict(state)
    #         target_f[0][action] = target
    #         self._model.fit(state, target_f, epochs=1, verbose=0)
    #     if self._epsilon > self._epsilon_min:
    #         self._epsilon *= self._epsilon_decay

    def replay(self):
        batch_size = self._batch_size
        if len(self._memory) < batch_size: 
            return
        samples = random.sample(self._memory, batch_size)
        for sample in samples:
            state, action, reward, new_state, done = sample
            target = self._target_model.predict(state,verbose=0,use_multiprocessing=True,workers=4)
            if done:
                target[0][action] = reward
            else:
                Q_future = max(self._target_model.predict(new_state,verbose=0,use_multiprocessing=True,workers=4)[0])
                target[0][action] = reward + Q_future * self._gamma
            self._model.fit(state, target, epochs=1, verbose=0,use_multiprocessing=True,workers=4)

    def target_train(self,done):
        if done:
            self._target_train_counter += 1
        if self._target_train_counter >= self._target_train_interval:
            self._target_model.set_weights(self._model.get_weights())
            self._target_train_counter = 0

        
    
    def act(self, state):
        if np.random.rand() <= self._epsilon:
            print(f"{self._epsilon:1.4f} random ",end="")
            return np.random.randint(0,728)
        else:
            print(f"{self._epsilon:1.4f} predict ",end="")
            act_values = self._model.predict(state,verbose=0)
            return np.argmax(act_values[0])

    def epsilon_update(self):
        if self._epsilon > self._epsilon_min:
            self._epsilon *= self._epsilon_decay
    
    def save(self, name="model.h5"):
        self._target_model.save(name,save_format="h5")
        #save memory
        np.save("memory.npy", np.array(self._memory))


            

class TrainingEnv:
    def __init__(self,params: dict = {}):
        self._sim = MujocoHandler(params.get("env_path","environment.xml"))
        self._cam = RGBD_CamHandler(self._sim,size=params.get("image_size",600),windowed=False,fps=params.get("fps",30))
        self._cam.R = params.get("cam_R",np.array([ [0., 0., -1.], [0., 1., 0.],[-1., 0., 0.]]))
        self._cam.t = params.get("cam_t",np.array([[0.7],[ 0. ],[4. ]]))
        self._dqn_agent = DQN(params)
        self._episodes = params.get("episodes",100)
        self._max_duration = params.get("max_sim_time",6)
        self._z_height = params.get("z_height",0.5)
        self._last_init_state = None
        self._tracker = None
        self._floor_collision_threshold = params.get("floor_collision_threshold",0.3)

    def _aparent_state(self,normalised=False):
        self._tracker.update_track()
        self._tracker.cv2_show(True,True,True)
        pos = self._tracker.ball_position()
        vel = self._tracker.ball_velocity()
        arm = self._sim.get_arm_state()
        state = np.concatenate((pos,vel,arm))
        if normalised:
            normalised_state = state / np.array([10.8,5.8,2.25,10,10,10,3.22886,2.70526,2.26893,6.10865,2.26892802759,6.10865])
            return normalised_state
        else:
            return state
    def train(self):
        self._tracker = BallTracker(self._cam)
        try:
            for episode in range(self._episodes):

                self._last_init_state = list(self._sim.set_random_state())
                self._cam.reset()
                self._tracker.reset()
                self._tracker.update_track()
                while not self._tracker.is_tracking() :
                    while not self._cam.update_frame():
                        self._sim.step()
                    self._tracker.update_track()
                self._run_episode()
                print (f"Episode {episode} finished",self._sim.is_ball_in_target())


        finally:
            self._dqn_agent.save()
        
                
            


        
    def _run_episode(self):
        cur_state = self._aparent_state(normalised=True).reshape(1,12)
        while self._sim.time < self._max_duration:
            action = self._dqn_agent.act(cur_state)
            self._sim.take_action(action,0.5)
            while not self._cam.update_frame():
                self._sim.step()
            if not self._tracker.is_tracking():
                break
            new_state = self._aparent_state(normalised=True).reshape(1,12)
            reward,done = self._reward_n_done()
            self._dqn_agent.remember(cur_state, action, 
                reward, new_state, done)
            
            self._dqn_agent.replay()
            self._dqn_agent.target_train(done=done)
            print (f"Time: {self._sim.time} Reward: {reward} Done: {done}")
            cur_state = new_state
            if done:
                break
        self._dqn_agent.epsilon_update()

 
    def _intersect_point(self,init_state,z_height):
        t = np.roots([9.81/2,init_state[5],init_state[2]-z_height]).max()
        xt = init_state[0] + init_state[3]*t
        yt = init_state[1] + init_state[4]*t
        zt = 0.5
        return np.array([xt,yt,zt])

    def _reward_n_done(self):
        score = 0
        done = False

        state = self._sim.get_arm_state()
        if ((not -3.22 < state[0] < 3.22) 
            or (not -2.70 < state[1] < 0.61) 
            or (not -2.26 < state[2] < 2.68) 
            or (not -6.10 < state[3] < 6.10) 
            or (not -2.26 < state[4] < 2.26) 
            or (not -6.10 < state[5] < 6.10)):


            score -= 100
            done = True

        r= Rotation.from_matrix(self._sim.get_basket_orientation())
        basket_angle = r.as_euler('xyz',degrees=True)[0]
        score -= abs(basket_angle)




        target = self._intersect_point(self._last_init_state,self._z_height)
        pos = self._sim.get_basket_position()

        distance = np.linalg.norm(target-pos)
        score -= distance*100

        if distance < 0.1:
            score += 100

        z = pos[2]
        if z < self._floor_collision_threshold:
            score -= 100
            if z < self._floor_collision_threshold:
                done = True
        if self._sim.get_n_contacts() > 0:
            if self._sim.is_ball_in_target():
                done = True
                score += 100
            if self._sim.is_ball_on_floor():
                score -= 100
                done = True
        
        if self._sim.time > self._max_duration:
            done = True
       
        
        return score,done

    


        


        
            