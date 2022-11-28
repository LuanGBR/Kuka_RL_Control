import numpy as np
from collections import deque
import random
from scipy.spatial.transform import Rotation
from Vision import RGBD_CamHandler, BallTracker, BallLostException
from Simulation import MujocoHandler
from Utils import CV2renderer
from functools import wraps
from time import time
from tqdm.auto import tqdm
from torch import device, cuda
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
device = device("cuda" if cuda.is_available() else "cpu")
from collections import namedtuple
from itertools import count

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print('func:%r args:[%r, %r] took: %2.4f sec' % (f.__name__, args, kw, te-ts))
        return result
    return wrap






class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class dqn_train_model(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size,32 )
        self.fc2 = nn.Linear(32, 128)
        self.fc3 = nn.Linear(128, 512)
        self.fc4 = nn.Linear(512, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)
    
class dqn_target_model(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size,32 )
        self.fc2 = nn.Linear(32, 128)
        self.fc3 = nn.Linear(128, 512)
        self.fc4 = nn.Linear(512, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)



class DQN:
    def __init__(self,params):
        self.params = params
        self._observation_space = params.get("observation_space", 12)
        self._action_space = params.get("action_space", 729)
        self._memory = ReplayMemory(params.get("memory_size", 1000))
        self._gamma = params.get('gamma', 0.95)
        eps = params.get("epsilon", [params.get("epsilon_max",1.0), params.get("epsilon_min",0.03), params.get("epsilon_decay",200)])
        self._epsilon_max, self._epsilon_min, self._epsilon_decay = eps
        self._learning_rate = params.get('learning_rate', 0.001)
        self._tau = params.get('tau', 0.125)
        self._batch_size = params.get('batch_size', 32)
        self._policy_net = dqn_train_model(self._observation_space, self._action_space).to(device)
        self._target_net = dqn_target_model(self._observation_space, self._action_space).to(device)
        self._target_net.load_state_dict(self._policy_net.state_dict())
        self._target_net.eval()
        self._optimizer = optim.Adam(self._policy_net.parameters(), lr=self._learning_rate)
        self._step = 0
        self._target_update_gap = params.get('target_update_gap', 10)
    
    @property
    def step(self):
        return self._step
    
    @step.setter
    def step(self, value):
        self._step = value

    def _epsilon(self):
        return self._epsilon_min + (self._epsilon_max - self._epsilon_min) * np.exp(-1. * self.step / self._epsilon_decay)
    
    def act(self, state):
        sample = np.random.random()
        if sample > self._epsilon():
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                act = self._policy_net(state).max(1)[1].view(1, 1)
        else:
            act =  torch.tensor([[np.random.randint(self._action_space)]], device=device, dtype=torch.long)
        self.step += 1
        return act
    
    def remember(self, state, action, next_state,reward):
        self._memory.push(state, action, next_state, reward)
    
    def optimize_model(self):
        if len(self._memory) < self._batch_size:
            return
        transitions = self._memory.sample(self._batch_size)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self._policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self._batch_size, device=device)
        next_state_values[non_final_mask] = self._target_net(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self._gamma) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self._optimizer.zero_grad()
        loss.backward()
        for param in self._policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self._optimizer.step()
    
    def target_update(self,episode_step):
       if episode_step % self._target_update_gap == 0:
                    self._target_net.load_state_dict(self._policy_net.state_dict())
        

            

class TrainingEnv:
    def __init__(self,params: dict = {}):
        self._sim = MujocoHandler(params.get("env_path","environment.xml"))
        self._cam = RGBD_CamHandler(self._sim,size=params.get("image_size",600),windowed=False,fps=params.get("fps",30))
        self._cam.R = params.get("cam_R",np.array([ [0., 0., -1.], [0., 1., 0.],[-1., 0., 0.]]))
        self._cam.t = params.get("cam_t",np.array([[0.7],[ 0. ],[4. ]]))
        self._dqn_agent = DQN(params)
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
        self._terminal_rewards = []
    

    def _aparent_state(self,normalised=False):
        self._tracker.update_track()
        pos = self._tracker.ball_position()
        vel = self._tracker.ball_velocity()
        arm = self._sim.get_arm_state()
        state = np.concatenate((pos,vel,arm))
        if normalised:
            normalised_state = state / np.array([10.8,5.8,2.25,10,10,10,3.22886,2.70526,2.26893,6.10865,2.26892802759,6.10865])
            return torch.tensor(normalised_state, device=device, dtype=torch.float).view(1, -1)
        else:
            return torch.tensor(state, device=device, dtype=torch.float).view(1, -1)
    def train(self):
        for i_episode in tqdm(range(self._num_episodes),desc="Episodes"):
            # Initialize the environment and state
            self._sim.reset()
            self._last_init_state = self._sim.set_random_state()
            self._cam.reset()
            self._tracker.reset()
            while not self._tracker.is_tracking() :
                    while not self._cam.update_frame():
                        self._sim.step()
                    self._tracker.update_track()
                    self.renderer.render(self._episode_durations,self._terminal_rewards)
                    
            self._run_episode()
            
            
            
    def _run_episode(self):
        last_screen = self._aparent_state()
        current_screen = self._aparent_state()
        state = current_screen - last_screen
        for t in tqdm(count(),desc="Steps"):
            # Select and perform an action
            action = self._dqn_agent.act(state)
            self._sim.take_action(action.item(), self._velocity_factor*0)
            while not self._cam.update_frame():
                self._sim.step()
            try:
                self._tracker.update_track()
            except BallLostException:
                break
            self.renderer.render(self._episode_durations,self._terminal_rewards)         
            reward, done = self._reward_n_done()
            reward = torch.tensor([reward], device=device, dtype=torch.float)

            # Observe new state
            last_screen = current_screen
            current_screen = self._aparent_state()
            if not done:
                next_state = current_screen - last_screen
            else:
                next_state = None

            # Store the transition in memory
            self._dqn_agent.remember(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            self._dqn_agent.optimize_model()
            if done:
                self._episode_durations.append(t + 1)
                self._terminal_rewards.append(reward.item())
                # plot_durations()
                break

            # Update the target network, copying all weights and biases in DQN
            self._dqn_agent.target_update(t)
        return
    
    def _intersect_point(self,init_state,z_height):
        t = np.roots([-9.81/2,init_state[5],init_state[2]-z_height]).max()
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
            score -= 500
            done = True

        r= Rotation.from_matrix(self._sim.get_basket_orientation())
        basket_angles = r.as_euler('xyz',degrees=True)
        score -= abs(basket_angles[0]) + abs(basket_angles[1])




        target = self._intersect_point(self._last_init_state,self._z_height)
        pos = self._sim.get_basket_position()

        distance = np.linalg.norm(target-pos)
        score -= distance*200

        if distance < 0.1:
            score += 100

        z = pos[2]
        
        
        
        if self._sim.time > self._max_duration:
            done = True
        
        if z < self._floor_collision_threshold:
            score -= 500
            if z < self._floor_collision_threshold:
                done = True
        if self._sim.get_n_contacts() > 0:
            if self._sim.is_ball_on_floor():
                score += 0 if done else +50 
                done = True
            if self._sim.is_ball_in_target():
                done = True
                score += 300
            
       
        
        return score,done
