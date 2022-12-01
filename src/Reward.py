import numpy as np
from scipy.spatial.transform import Rotation

class Reward:
    def continous_barrier_penalty(self,x, x_min, x_max, C=1, max_penalty=1):
        if x_min < x < x_max:
            return np.min([max_penalty,C*(np.log(0.25*(x_max-x_min)*(x_max-x_min))-np.log((x-x_min)*(x_max-x)))])
        else:
            return max_penalty
        
    def hyperbolic_penalty(self,x,xmin,xmax,lamb,tau):
        x_minus_xmin = x - xmin
        xmax_minus_x = xmax - x

        tau_squared = tau*tau
        lamb_squared = lamb*lamb

        p = -lamb*x_minus_xmin+ np.sqrt(lamb_squared*x_minus_xmin * x_minus_xmin+tau_squared)-lamb*xmax_minus_x + np.sqrt(lamb_squared*xmax_minus_x*xmax_minus_x+tau_squared)
        return p

    def _intersect_point(self,init_state,z_height):
        t = np.roots([-9.81/2,init_state[5],init_state[2]-z_height]).max()
        xt = init_state[0] + init_state[3]*t
        yt = init_state[1] + init_state[4]*t
        zt = z_height
        p = np.array([xt,yt,zt])
        return p

    def _reward_n_done(self):
        #done
        state = self._sim.get_arm_state()
        pos = self._sim.get_basket_position()
        z = pos[2]

        done = False

        if ((not -3.22 < state[0] < 3.22)
            or (not -2.70 < state[1] < 0.61)
            or (not -2.26 < state[2] < 2.68)
            or (not -6.10 < state[3] < 6.10)
            or (not -2.26 < state[4] < 2.26)
            or (not -6.10 < state[5] < 6.10)):
            done = True
        
        if self._sim.time > self._max_duration:
            done = True
        
        if self._sim.get_n_contacts() > 0:
            if self._sim.is_ball_on_floor():
                done = True
            if self._sim.is_ball_in_target():
                
                done = True
        
        if z<self._floor_collision_threshold:
            done = True
        
        #reward
        score = 0

        target = self._intersect_point(self._last_init_state,self._z_height)
        phi = np.arctan2(target[1],target[0])
        
        dist = np.linalg.norm(pos - target) 
        score -= dist #distance to target penalty

        angular_penalty = 2*self.hyperbolic_penalty(state[0],phi-0.2,phi+0.2,1,0.1)
        score -= angular_penalty #angular penalty to fasten learning

        r= Rotation.from_matrix(self._sim.get_basket_orientation())
        basket_angles = r.as_euler('xyz')
        score -= abs(basket_angles[0]) + abs(basket_angles[1]) #basket orientation penalty

        score -= self.continous_barrier_penalty(z,self._floor_collision_threshold,2*self._z_height-self._floor_collision_threshold,0.025,1) #continous barrier log penalty z height

        #BONUS FOR FINISH 
        if self._sim.get_n_contacts() > 0:
            if self._sim.is_ball_in_target():
                score += 10

        ball_state = self._sim.get_ball_state()
        ball_pos = ball_state[0:3]
        pos2d  = pos[0:2]
        ball_pos2d = ball_pos[0:2]
        target2d = target[0:2]

        #race to the target
        vec_target_ball = target2d - ball_pos2d #vector from ball to target
        vec_basket_ball = pos2d - ball_pos2d #vector from ball to basket
        vec_target_basket = target2d - pos2d #vector from basket to target
        norm_target_ball = np.linalg.norm(vec_target_ball) #norm of the vector from target to ball
        target_ball_unit = vec_target_ball/norm_target_ball #unit vector from ball to target

        a = np.dot(target_ball_unit,vec_basket_ball) #projection of the vector from ball to basket on the unit vector from ball to target
        b = np.dot(target_ball_unit,vec_target_basket) #projection of the vector from basket to target on the unit vector from ball to target
        score += 2*(a - b) #race to the target bonus 

        return score,done

    