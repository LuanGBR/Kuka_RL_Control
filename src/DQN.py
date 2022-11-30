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

class ReplayBuffer:
    """Fixed -size buffe to store experience tuples."""
    def __init__(self, action_size, buffer_size, batch_size, seed=random.random()):
        """Initialize a ReplayBuffer object.
        
        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        
        self._action_size = action_size
        self._memory = deque(maxlen=buffer_size)
        self._batch_size = batch_size
        self._experiences = namedtuple("Experience", field_names=["state",
                                                               "action",
                                                               "reward",
                                                               "next_state",
                                                               "done"])
        self.seed = random.seed(seed)
        
    def push(self,state, action, next_state,reward,done):
        """Add a new experience to memory."""
        e = self._experiences(state,action,reward,next_state,done)
        self._memory.append(e)
        
    def sample(self):
        """Randomly sample a batch of experiences from memory"""
        experiences = random.sample(self._memory,k=self._batch_size)
        
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e.next_state is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e.next_state is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e.next_state is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e.next_state is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e.next_state is not None]).astype(np.uint8)).float().to(device)
        
        return (states,actions,rewards,next_states,dones)
    def __len__(self):
        """Return the current size of internal memory."""
        return len(self._memory)
    
    
    def load(self, path="saved_data/memory.pth"):
        self._memory = torch.load(path)[:self._capacity]
        self._memory = deque(self._memory, maxlen=self._capacity)

class DQN_model(nn.Module):

    def __init__(self, input_size, output_size):
        super(DQN_model, self).__init__()
        self.fc1 = nn.Linear(input_size,2048)
        self.fc2 = nn.Linear(2048,2048)
        self.fc3 = nn.Linear(2048,2048)
        self.fc4 = nn.Linear(2048, output_size)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = self.fc4(x)
        return x
    
    def optimizer(self, lr):
        return optim.Adam(self.parameters(), lr=lr)
    
    def loss(self):
         return torch.nn.MSELoss().to(device)


######################################################################
class DQN_Agent:
    def __init__(self,params):
        # Read parameters
        self.params = params
        self._observation_space = params.get("observation_space", 15)
        self._action_space = params.get("action_space", 729)
        
        self._gamma = params.get('gamma', 0.999)
        eps = params.get("epsilon", [params.get("epsilon_max",1.0), params.get("epsilon_min",0.03), params.get("epsilon_decay",0.99)])
        self._epsilon, self._epsilon_min, self._epsilon_decay = eps
        self._tau = params.get("tau",0.001)
        self._learning_rate = params.get('learning_rate', 0.001)
        self._batch_size = params.get('batch_size', 128)
        self._memory = ReplayBuffer(buffer_size=params.get("memory_size", 10000),action_size=self._observation_space,batch_size=self._batch_size,seed=0)
        self.update_gap = params.get('update_gap', 10)

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
        self.update_counter = 0

    def epsilon_update(self):
        self._epsilon = max(self._epsilon*self._epsilon_decay,self._epsilon_min) # decays epsilon, if not less than epsilon minimum



    def select_action(self, state):
        self._step += 1
        # Select action based on current policy
        rng = random.random()
        eps = self._epsilon
        if rng < eps:
            action = random.randint(0,728)
        else:
            with torch.no_grad():
                state = torch.tensor(state,device=device,dtype=torch.float32).unsqueeze(0)
                action = self._policy_net(state).max(1)[1].item()
        return action
    
    def remember(self, state,action,next_state,reward,done):
        self._memory.push(state, action, next_state, reward,done)
    
    def save(self,target_path="saved_data/target.pth",policy_path="saved_data/policy.pth"):
        torch.save(self._policy_net.state_dict(), policy_path)
        torch.save(self._target_net.state_dict(), target_path)
    
    def load(self,target_path="saved_data/target.pth",policy_path="saved_data/policy.pth"):
        self._policy_net.load_state_dict(torch.load(policy_path))
        self._target_net.load_state_dict(torch.load(target_path))
        self._memory.load()
        
    
        



        # EPS_DECAY = 200
        # TARGET_UPDATE = 10

    def optimize_model(self):

        self.update_counter = (self.update_counter+1)% self.update_gap
        if self.update_counter != 0:
            return

        if len(self._memory) < self._batch_size:
            return


        states, actions, rewards, next_states, dones = self._memory.sample()
        
        ## TODO: compute and minimize the loss
        criterion = self._loss
        # Local model is one which we need to train so it's in training mode
        self._policy_net.train()
        # Target model is one with which we need to get our target so it's in evaluation mode
        # So that when we do a forward pass with target model it does not calculate gradient.
        # We will update target model weights with soft_update function

        #shape of output from the model (batch_size,action_dim) = (64,4)
        predicted_targets = self._policy_net(states).gather(1,actions)

    
    
        with torch.no_grad():
            labels_next = self._target_net(next_states).detach().max(1)[0].unsqueeze(1)

        # .detach() ->  Returns a new Tensor, detached from the current graph.
        labels = rewards + (self._gamma* labels_next*(1-dones))
        
        loss = criterion(predicted_targets,labels).to(device)
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
        

        # ------------------- update target network ------------------- #
        self.soft_update()
        self.epsilon_update()

    def soft_update(self):
        for target_param, local_param in zip(self._target_net.parameters(),
                                           self._policy_net.parameters()):
            target_param.data.copy_(self._tau*local_param.data + (1-self._tau)*target_param.data)


