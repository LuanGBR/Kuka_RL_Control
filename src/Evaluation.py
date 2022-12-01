
from .Run import Run
from tqdm import trange
import numpy as np
class EvaluationEnv(Run):
    def __init__(self,model_filepath="saved_data/target.pth", params={}):
        super(EvaluationEnv,self).__init__(params)
        self._dqn_agent.load(model_filepath)
        self._results = []
    
    def evaluate(self,n=100):
        print('Evaluating...')
        t = trange(n,desc=f"Success rate: {np.round(np.mean(self._results),2)}")
        for i in t:
            self.reset()
            self.wait_for_ball()
            self._run_episode()
            srate = np.sum(self._results)/len(self._results)
            t.set_description(f"Success rate: {np.round(srate,2)}", refresh=True)
    
    def _run_episode(self):
        state = self._aparent_state()
        for t in range(1000):
            # Select and perform an action
            action = self._dqn_agent.select_action(state,eval_mode=True)
            _, done = self.step(action)
            # Observe new state
            next_state =  None if done else self._aparent_state()

            # Move to the next state
            state = next_state

            if done:
                self._results.append(self._sim.is_ball_in_target())
                break

    def reset(self):
        self._sim.reset()
        self._last_init_state = self._sim.set_random_state()
        self._cam.reset()
        self._tracker.reset()

    


    
        

        