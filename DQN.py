from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import numpy as np
import random

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
        self._memory = deque(maxlen=params.get('memory_size', 1000))
        self._gamma = params.get('gamma', 0.95)
        self._epsilon = params.get('epsilon', 1.0)
        self._epsilon_decay = params.get('epsilon_decay', 0.995)
        self._epsilon_min = params.get('epsilon_min', 0.01)
        self._learning_rate = params.get('learning_rate', 0.001)
        self._tau = params.get('tau', 0.125)
        self._batch_size = params.get('batch_size', 32)
        self._model = self._create_model()
        self._target_model = self._create_model()


    def _create_model(self):
        model   = Sequential()
        state_shape  = self.env.observation_space.shape # number of state variables
        model.add(Dense(24, input_dim=state_shape[0], 
            activation="relu"))
        model.add(Dense(48, activation="relu"))
        model.add(Dense(24, activation="relu"))
        model.add(Dense(self.env.action_space.n)) #number of actions
        model.compile(loss="mean_squared_error",
            optimizer=Adam(lr=self.learning_rate))
        return model

    def _remember(self, state, action, reward, next_state, done):
        self._memory.append((state, action, reward, next_state, done))

    def _replay(self):
        if len(self._memory) < self._batch_size:
            return
        minibatch = random.sample(self._memory, self._batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self._gamma * 
                    np.amax(self._target_model.predict(next_state)[0]))
            target_f = self._model.predict(state)
            target_f[0][action] = target
            self._model.fit(state, target_f, epochs=1, verbose=0)
        if self._epsilon > self._epsilon_min:
            self._epsilon *= self._epsilon_decay

    def _target_train(self):
        weights = self._model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i]
        self._target_model.set_weights(target_weights)
    
    def _act(self, state):
        if np.random.rand() <= self._epsilon:
            return self.env.action_space.sample()
        act_values = self._model.predict(state)
        return np.argmax(act_values[0])


        




