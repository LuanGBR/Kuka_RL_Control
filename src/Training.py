import numpy as np
from collections import deque
import random
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from scipy.spatial.transform import Rotation
import tensorflow as tf
from keras.callbacks import TensorBoard
from Vision import RGBD_CamHandler, BallTracker
from Simulation import MujocoHandler


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