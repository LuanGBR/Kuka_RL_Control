import numpy as np
from collections import deque
from scipy.spatial.transform import Rotation
from .Vision import RGBD_CamHandler, BallTracker, BallLostException
from .Simulation import MujocoHandler
from .Utils import CV2renderer,Plots

from tqdm.auto import trange
from torch import device, cuda
import torch
device = device("cuda" if cuda.is_available() else "cpu")
from collections import namedtuple
from itertools import count
import matplotlib.pyplot as plt
from .DQN import DQN_Agent

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


def continous_barrier_penalty(x, x_min, x_max, C=1, max_penalty=1):
    if x_min < x < x_max:
        return np.min([max_penalty,C*(np.log(0.25*(x_max-x_min)*(x_max-x_min))-np.log((x-x_min)*(x_max-x)))])
    else:
        return max_penalty
    
def hyperbolic_penalty(x,xmin,xmax,lamb,tau):
    x_minus_xmin = x - xmin
    xmax_minus_x = xmax - x

    tau_squared = tau*tau
    lamb_squared = lamb*lamb

    p = -lamb*x_minus_xmin+ np.sqrt(lamb_squared*x_minus_xmin * x_minus_xmin+tau_squared)-lamb*xmax_minus_x + np.sqrt(lamb_squared*xmax_minus_x*xmax_minus_x+tau_squared)
    return p
     

class TrainingEnv:
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
    

    def _aparent_state(self,normalised=True):
        pos = self._tracker.ball_position()
        vel = self._tracker.ball_velocity()
        arm = self._sim.get_arm_state()
        basket_pos = self._sim.get_basket_position()
        state = np.concatenate((pos,vel,arm,basket_pos))
        if normalised:
            normalised_state = state / np.array([10.8,5.8,2.25,10,10,10,3.22886,2.70526,2.26893,6.10865,2.26892802759,6.10865,10.8,5.8,2.25])
            return torch.tensor(normalised_state, device=device, dtype=torch.float).view(1, -1)
        else:
            return torch.tensor(state, device=device, dtype=torch.float).view(1, -1)
    def train(self):
        try:
            t = trange(self._num_episodes, desc='Episode', leave=True)
            for i_episode in t:
                t.set_description(f"Episode {i_episode}, rwd: {np.round(self._terminal_rewards[-1],4) if self._terminal_rewards else '--'} eps: {self._dqn_agent._epsilon}", refresh=True)
                # Initialize the environment and state
                self.reset()
                self.wait_for_ball()
                self._run_episode()
                self._dqn_agent.target_update()
                if i_episode % 10 == 0 and i_episode > 0:
                    Plots.live_plot(self._terminal_rewards)
                
            
        finally:
            print('\nComplete')
            self._dqn_agent.save()
        
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
            


    

            
            
            
    def _run_episode(self):
      
        state = self._aparent_state()
        for t in count():
            # Select and perform an action
            action = self._dqn_agent.select_action(state)
            try:
                reward, done = self.step(action)
            except BallLostException:
                break
            
            reward = torch.tensor([reward], device=device, dtype=torch.float)

            # Observe new state
            next_state =  None if done else self._aparent_state()
            # Store the transition in memory
            self._dqn_agent.remember(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            self._dqn_agent.optimize_model()
            if done:
                break
        self._terminal_rewards.append(reward.item())
    
    def _intersect_point(self,init_state,z_height):
        t = np.roots([-9.81/2,init_state[5],init_state[2]-z_height]).max()
        xt = init_state[0] + init_state[3]*t
        yt = init_state[1] + init_state[4]*t
        zt = z_height
        p = np.array([xt,yt,zt])
        return p

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
            done = True

        score -= continous_barrier_penalty(state[0],-3.22,3.22,C=0.025,max_penalty=1)
        score -= continous_barrier_penalty(state[1],-2.70,0.61,C=0.025,max_penalty=1)
        score -= continous_barrier_penalty(state[2],-2.26,2.68,C=0.025,max_penalty=1)
        score -= continous_barrier_penalty(state[3],-6.10,6.10,C=0.025,max_penalty=1)
        score -= continous_barrier_penalty(state[4],-2.26,2.26,C=0.025,max_penalty=1)
        score -= continous_barrier_penalty(state[5],-6.10,6.10,C=0.025,max_penalty=1)




        # r= Rotation.from_matrix(self._sim.get_basket_orientation())
        # basket_angles = r.as_euler('xyz',degrees=True)
        # score -= abs(basket_angles[0])/180 + abs(basket_angles[1])/180




        target = self._intersect_point(self._last_init_state,self._z_height)
        pos = self._sim.get_basket_position()

        score -= np.abs(pos[0] - target[0])/10.8
        score -= np.abs(pos[1] - target[1])/5.8
        score -= np.abs(pos[2] - target[2])/2.8



        z = pos[2]
        
        
        
        if self._sim.time > self._max_duration:
            done = True
        
        
            
        if self._sim.get_n_contacts() > 0:
            if self._sim.is_ball_on_floor():
                score += 0 if done else +0 
                done = True
            if self._sim.is_ball_in_target():
                done = True
                score +12
        

        #continous barrier log penalty z height
        score -= continous_barrier_penalty(z,self._floor_collision_threshold,2*self._z_height-self._floor_collision_threshold,0.025,1)
        
        if z<self._floor_collision_threshold:
            score -= 1
            done = True

        
        score = score/13
            
       
        
        return score,done



    def _reward_n_done(self):
        #done
        state = self._sim.get_arm_state()
        pos = self._sim.get_basket_position()
        z = pos[2]

        done = False

        if ((not -3.22 < state[0] < 3.22)
            or (not -2.70 < state[1] < 0.61)
            or (not -2.26 < state[2] < 2.68)
            or (not -6.10 < state[3] < 6.10)
            or (not -2.26 < state[4] < 2.26)
            or (not -6.10 < state[5] < 6.10)):
            done = True
        
        if self._sim.time > self._max_duration:
            done = True
        
        if self._sim.get_n_contacts() > 0:
            if self._sim.is_ball_on_floor():
                done = True
            if self._sim.is_ball_in_target():
                done = True
        
        if z<self._floor_collision_threshold:
            done = True
        
        #reward
        score = 0

        target = self._intersect_point(self._last_init_state,self._z_height)
        phi = np.arctan2(target[1],target[0])


        # score -= np.linalg.norm(pos - target)

        angular_penalty = 2*hyperbolic_penalty(state[0],phi-0.2,phi+0.2,1,0.1)
        score -= angular_penalty




        return score,done

        
