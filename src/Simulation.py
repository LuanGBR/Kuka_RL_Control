import mujoco
import numpy as np


class MujocoHandler:
    """A handler for the mujoco environment."""
    def __init__(self,model_path):
        model = mujoco.MjModel.from_xml_path(model_path)
        data = mujoco.MjData(model)
        self._model = model
        self._data = data
        self._cam_aux_count = 0
        self._max_velocities = np.array([2.72271363311,2.72271363311,5.75958653158,5.75958653158,5.75958653158,10.7337748998]).astype(np.float32)

    @property
    def model(self):
        return self._model
    
    @property
    def data(self):
        return self._data
    
    @property
    def time(self):
        return self._data.time
    
    @property
    def timestep(self):
        return self._model.opt.timestep
    
    @property
    def ticks(self):
        return self._data.solver_iter
    
    def step(self):
        self._cam_aux_count += 1
        mujoco.mj_step(self._model, self._data)
    
    def set_state(self, init_pos, init_vel):
        self._model.qpos0[0:3] = init_pos
        self.reset()
        self._data.qvel[0:3] = init_vel
        mujoco.mj_forward(self._model, self._data)
    
    def reset(self):
        mujoco.mj_resetData(self._model, self._data)
        mujoco.mj_forward(self._model, self._data)
    
    def set_random_state(self):
        init_pos = np.array([-0.966,-2,1]) 
        V_abs = np.random.uniform(4.9,5.1)
        # V_theta = np.random.uniform(np.pi/4,np.pi/3) # TODO: check if this is the right range
        V_theta = np.pi/4
        vx = 0
        vy = V_abs*np.cos(V_theta)
        vz = V_abs*np.sin(V_theta)
        init_vel = np.array([vx,vy,vz])

        self.set_state(init_pos, init_vel)
        return np.concatenate((init_pos, init_vel))

    def get_ball_state(self):
        return np.concatenate((self._data.body("ball").xpos, self._data.body("ball").cvel[3:6]))
    
    def get_arm_state(self):
        return (self._data.qpos[7:13])

    def get_ball(self):
        return self._env.data.body("ball")

    def take_action(self,action,k):
        ternary = np.base_repr(action, base=3)
        ternary = ternary.zfill(6)


        for i,n in enumerate(ternary):
            self._data.ctrl[i] = (int(n) - 1) * k * self._max_velocities[i]
        pass
    
    def get_n_contacts(self):
        return self._data.ncon
    
    def get_body_contacts(self):
        body_pairs =  [(self._model.body(self._model.geom(contact.geom1).bodyid[0]).name,self._model.body(self._model.geom(contact.geom2).bodyid[0]).name) for contact in self._data.contact]
        return body_pairs
    def is_ball_in_target(self):
        body_pairs = self.get_body_contacts()
        if ("ball","target") in body_pairs or ("target","ball") in body_pairs:
            return True
        else:
            return False
    
    def is_ball_on_floor(self):
        body_pairs = self.get_body_contacts()
        if ("ball","world") in body_pairs or ("world","ball") in body_pairs:
            return True
        else:
            return False
    
    def get_basket_position(self):
        return self._data.body("target").xpos
    
    def get_basket_orientation(self):
        return self._data.body("target").xmat.reshape((3,3))