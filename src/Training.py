import numpy as np
from tqdm.auto import trange
from collections import namedtuple
from itertools import count
from .Run import Run
from .Utils import Plots
from .Vision import BallLostException


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class TrainingEnv(Run):
    def __init__(self,params={}):
        super(TrainingEnv,self).__init__(params)

    def train(self):
        try:
            t = trange(self._num_episodes, desc='Episode', leave=True)
            for i_episode in t:
                t.set_description(f"Episode {i_episode}, rwd: {np.round(self._terminal_rewards[-1],4) if self._terminal_rewards else '--'} eps: {self._dqn_agent._epsilon}", refresh=True)
                # Initialize the environment and state
                self.reset()
                self.wait_for_ball()
                self._run_episode()
                self._dqn_agent.soft_update()
                if i_episode % 10 == 0 and i_episode > 0:
                    Plots.live_plot(self._terminal_rewards)
        finally:
            print('\nComplete')
            self._dqn_agent.save()

    def _run_episode(self):
      
        state = self._aparent_state()
        for t in count():
            # Select and perform an action
            action = self._dqn_agent.select_action(state)
            try:
                reward, done = self.step(action)
            except BallLostException:
                break
            
            # Observe new state
            next_state =  None if done else self._aparent_state()
            # Store the transition in memory
            self._dqn_agent.remember(state, action, next_state, reward,done)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            self._dqn_agent.optimize_model()
            if done:
                break
        self._terminal_rewards.append(reward.item())
    
    

        
