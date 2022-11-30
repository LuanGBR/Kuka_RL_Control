import random
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple, deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self._capacity = capacity
        self._memory = deque([],maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self._memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self._memory, batch_size)

    def __len__(self):
        return len(self._memory)
    
    def save(self, path="saved_data/memory.pth"):
        torch.save(self._memory, path)
    
    def load(self, path="saved_data/memory.pth"):
        self._memory = torch.load(path)[:self._capacity]
        self._memory = deque(self._memory, maxlen=self._capacity)

class DQN_model(nn.Module):

    def __init__(self, input_size, output_size):
        super(DQN_model, self).__init__()
        self.fc1 = nn.Linear(input_size,128)
        self.fc3 = nn.Linear(128,output_size)     

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = self.fc3(x)
        return x
    
    def optimizer(self, lr):
        return optim.Adam(self.parameters(), lr=lr)
    
    def loss(self):
         return nn.SmoothL1Loss().to(device)


######################################################################
class DQN_Agent:
    def __init__(self,params):
        # Read parameters
        self.params = params
        self._observation_space = params.get("observation_space", 15)
        self._action_space = params.get("action_space", 729)
        self._memory = ReplayMemory(params.get("memory_size", 10000))
        self._gamma = params.get('gamma', 0.999)
        eps = params.get("epsilon", [params.get("epsilon_max",1.0), params.get("epsilon_min",0.03), params.get("epsilon_decay",0.99)])
        self._epsilon, self._epsilon_min, self._epsilon_decay = eps
        self._learning_rate = params.get('learning_rate', 0.001)
        self._batch_size = params.get('batch_size', 128)
        self._target_update_gap = params.get('target_update_gap', 10)

        # Initialize model
        self._policy_net = DQN_model(self._observation_space, self._action_space).to(device)
        self._target_net = DQN_model(self._observation_space, self._action_space).to(device)

        # Initialize optimizer
        self._optimizer = self._policy_net.optimizer(self._learning_rate)

        # Get loss function
        self._loss = self._policy_net.loss()

        #
        self._target_net.load_state_dict(self._policy_net.state_dict()) # copy weights
        self._target_net.eval() # set to evaluation mode, gradient not computed
        
        # Initialize counters
        self._step = 0
        self._target_update_counter = 0

    def epsilon_update(self):
        self._epsilon = max(self._epsilon*self._epsilon_decay,self._epsilon_min) # decays epsilon, if not less than epsilon minimum



    def select_action(self, state):
        self._step += 1
        # Select action based on current policy
        rng = random.random()
        eps = self._epsilon
        if rng < eps:
            action = random.randint(0,728)
            action = torch.tensor([[action]],device=device,dtype=torch.int64)
        else:
            with torch.no_grad():
                action = self._policy_net(state).max(1)[1].view(1, 1)
        return action
    
    def remember(self, state,action,next_state,reward):
        self._memory.push(state, action, next_state, reward)
    
    def save(self,target_path="saved_data/target.pth",policy_path="saved_data/policy.pth"):
        torch.save(self._policy_net.state_dict(), policy_path)
        torch.save(self._target_net.state_dict(), target_path)
        self._memory.save()
    
    def load(self,target_path="saved_data/target.pth",policy_path="saved_data/policy.pth"):
        self._policy_net.load_state_dict(torch.load(policy_path))
        self._target_net.load_state_dict(torch.load(target_path))
        self._memory.load()
        
    
        



        # EPS_DECAY = 200
        # TARGET_UPDATE = 10

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
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
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
        criterion = self._loss
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self._optimizer.zero_grad()
        loss.backward()
        for param in self._policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self._optimizer.step()
        self.epsilon_update()

    def target_update(self):
        self._target_update_counter +=1
        if self._target_update_counter > self._target_update_gap:
            self._target_update_counter=0
            self._target_net.load_state_dict(self._policy_net.state_dict())


