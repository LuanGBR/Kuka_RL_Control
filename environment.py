import mujoco
import mujoco_viewer


class MujocoHandler:
    def __init__(self, model, data):
        self._model = model
        self._data = data

    @property
    def model(self):
        return self._model
    
    @property
    def data(self):
        return self._data
    
    def step(self):
        mujoco.mj_step(self.model, self.data)