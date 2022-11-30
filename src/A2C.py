"""
Definition of all agents
"""

import random
import pickle
import numpy as np
from collections import defaultdict, deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

"""
Training Hyperparameters
"""

ALPHA = 0.01
GAMMA = 0.99
LR = 3E-4

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

class ActorCritic(nn.Module):
	def __init__(self, in_dim,out_dim,hidden=2048):
		super(ActorCritic, self).__init__()

		self.critic_linear1 = nn.Linear(in_dim, hidden)
		self.critic_linear2 = nn.Linear(hidden, 1)

		self.actor_linear1 = nn.Linear(in_dim, hidden)
		self.actor_linear2 = nn.Linear(hidden, out_dim)

	def forward(self, state):
		state = Variable(torch.from_numpy(state).float().unsqueeze(0))
		value = F.relu(self.critic_linear1(state))
		value = self.critic_linear2(value)
		
		policy_dist = F.relu(self.actor_linear1(state))
		policy_dist = F.softmax(self.actor_linear2(policy_dist), dim=1)

		return value, policy_dist


	


class A2C():
	def __init__(self,
				n_actions,
				observation_space,
				lr = LR,
				file_name="SavedA2C",
				gamma=GAMMA,
				max_steps=300,
				hidden=128,
				network='linear',device='cpu',epsilon=0.9,epsilon_decay=0.99,epsilon_min=0.01):
		self.name = "A2C"
		print("Inits A2C Agent")
		self.device = device
		print("Using device:",self.device)
		self.file_name = file_name
		self.gamma = gamma
		self.n_actions = n_actions
		self.max_steps = max_steps
		self.entropy_term = 0
		self.n_steps = 0
		self.all_n_steps = np.zeros((100))
		self.curr = 0
		self.log_probs = []
		self.values = []
		self.rewards = []
		self.epsilon = epsilon
		self.epsilon_decay = epsilon_decay
		self.epsilon_min = epsilon_min

		if network == 'linear':
			self.actor_critic = ActorCritic(observation_space,n_actions,hidden=hidden)

		elif network == 'conv': 
			print("Conv Net not implemented")
			pass

		self.ac_optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr)

	def choose_random_action(self, state):
		"""
		Choose an action according to the policy
		
		Parameters
		----------
		state
			Agent's state
		
		Returns
		-------
		action
			Action choosen
		"""

		value, policy_dist = self.actor_critic.forward(state)
		value = value.detach().numpy()[0,0]
		dist = policy_dist.detach().numpy()
		action = np.random.choice(self.n_actions, p=np.squeeze(dist)) 
		index= action

		log_prob = torch.log(policy_dist.squeeze(0)[index])
		entropy = -np.sum(np.mean(dist) * np.log(dist))

		self.values.append(value)
		self.log_probs.append(log_prob)
		self.entropy_term += entropy

		return action

	def choose_best_action(self, state):
		"""
		Choose an action according to the policy
		
		Parameters
		----------
		state
			Agent's state
		
		Returns
		-------
		action
			Action choosen
		"""

		value, policy_dist = self.actor_critic.forward(state)
		value = value.detach().numpy()[0,0]
		dist = policy_dist.detach().numpy()
		action = np.argmax(dist)

		log_prob = torch.log(policy_dist.squeeze(0)[action])
		entropy = -np.sum(np.mean(dist) * np.log(dist))

		self.values.append(value)
		self.log_probs.append(log_prob)
		self.entropy_term += entropy

		return action
	
	def choose_action(self, state):
		self.epsilon = max(self.epsilon*self.epsilon_decay,self.epsilon_min)
		if random.random() < self.epsilon:
			action = self.choose_best_action(state)
		else:
			action = self.choose_random_action(state)
		
		return action
		

		

	def update(self, 
		 	  state,
		 	  action,
		 	  next_state,
              reward,
		 	  done,
		 	  batch_size=128,
		 	  epochs=1,
		 	  save=True):

		self.n_steps += 1
		self.rewards.append(reward)
		
		if done or self.n_steps == self.max_steps:
			Qval, _ = self.actor_critic.forward(next_state)
			Qval = Qval.detach().numpy()[0,0]

			# compute Q values
			Qvals = np.zeros_like(self.values)
			for t in reversed(range(len(self.rewards))):
				Qval = self.rewards[t] + self.gamma * Qval
				Qvals[t] = Qval
	  
			#update actor critic
			values = torch.FloatTensor(self.values)
			Qvals = torch.FloatTensor(Qvals)
			log_probs = torch.stack(self.log_probs)
			
			advantage = Qvals - values
			actor_loss = (-log_probs * advantage).mean()
			critic_loss = 0.5 * advantage.pow(2).mean()
			ac_loss = actor_loss + critic_loss + 0.001 * self.entropy_term

			self.ac_optimizer.zero_grad()
			ac_loss.backward()
			self.ac_optimizer.step()

			self.curr += 1
			if self.curr >= 100:
				self.curr = 0
			self.all_n_steps[self.curr] = self.n_steps
			self.n_steps = 0
			self.log_probs = []
			self.values = []
			self.rewards = []
			del values
			del Qvals
			del log_probs	

	def save_train(self,folder_name="Saved/"):
		"""
		Save NN's parameters
		"""
		path = folder_name+self.file_name
		torch.save(self.actor_critic.state_dict(), path)

	def load_train(self,folder_name="Saved/"):
		"""
		Load NN's parameters into agent
		"""
		path = folder_name+self.file_name
		try:
			self.actor_critic.load_state_dict(torch.load(path))
			print("Train Loaded")
		except FileNotFoundError:
			print("Saved Train not found, continuing without loading")

		except EOFError:
			print("Saved Train not completed, continuing without loading")