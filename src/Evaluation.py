
class Evaluation:
    def __init__(self,env,agent):
        self._env = env
        self._agent = agent
        self._num_episodes = 10000
    
    def evaluate(self):
        for i in range(self._num_episodes):
            self._env.set_random_state()
            self._agent.reset()
            done = False
            while not done:
                action = self._agent.get_action(self._env.get_state())
                self._env.step(action)
                done = self._env.is_done()
            print("Episode: {}, Score: {}".format(i,self._env.get_score()))
            self._agent.save_model()