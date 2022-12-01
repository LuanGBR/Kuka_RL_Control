import numpy as np
from .Vision import RGBD_CamHandler, BallTracker
from .Simulation import MujocoHandler
from .Utils import CV2renderer
from collections import deque
import matplotlib.pyplot as plt
from .DQN import DQN_Agent
from .Reward import Reward
class Run(Reward):
    def __init__(self,params: dict = {}):
        self._sim = MujocoHandler(params.get("env_path","environment.xml"))
        self._cam = RGBD_CamHandler(self._sim,size=params.get("image_size",600),windowed=False,fps=params.get("fps",30))
        self._cam.R = params.get("cam_R",np.array([ [0., 0., -1.], [0., 1., 0.],[-1., 0., 0.]]))
        self._cam.t = params.get("cam_t",np.array([[0.7],[ 0. ],[4. ]]))
        self._dqn_agent = DQN_Agent(params)
        self._num_episodes = params.get("num_episodes",100)
        self._max_duration = params.get("max_sim_time",6)
        self._z_height = params.get("z_height",0.5)
        self._velocity_factor = params.get("velocity_factor",0.5)
        self._last_init_state = None
        self._tracker = BallTracker(self._cam)
        self._floor_collision_threshold = params.get("floor_collision_threshold",0.3)
        self._episode_durations = []
        self._render = params.get("render",True)
        if self._render:
            self.renderer = CV2renderer(cam=self._cam,tracker=self._tracker)
        self._plot = params.get("plot",False)
        self._terminal_rewards = deque([], maxlen=2000)
        
    def reset(self):
        self._sim.reset()
        self._last_init_state = self._sim.set_random_state()
        self._cam.reset()
        self._tracker.reset()

    def wait_for_ball(self):
        while not self._tracker.is_tracking() :
            while not self._cam.update_frame():
                self._sim.step()
            self._tracker.update_track()

    def step(self,action):
        self._sim.take_action(action, self._velocity_factor)
        while not self._cam.update_frame():
            self._sim.step()
        self._tracker.update_track()
        if self._render:
            self.renderer.render(self._episode_durations,self._terminal_rewards)
        return self._reward_n_done()
    
    def _aparent_state(self,normalised=True):
        pos = self._tracker.ball_position()
        vel = self._tracker.ball_velocity()
        arm = self._sim.get_arm_state()
        basket_pos = self._sim.get_basket_position()
        state = np.concatenate((pos,vel,arm,basket_pos))
        if normalised:
            normalised_state = state / np.array([10.8,5.8,2.25,10,10,10,3.22886,2.70526,2.26893,6.10865,2.26892802759,6.10865,10.8,5.8,2.25])
            return normalised_state
        else:
            return state