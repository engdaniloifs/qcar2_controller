import numpy as np
import time
from autonomy.GMPC.enum_class import  ControllerType

class FBLinearizationController:
    def __init__(self, Kp=np.array([2, 2, 2, 2])):
        self.controllerType = ControllerType.FEEDBACK_LINEARIZATION
        k1, k2, k3, k4 = Kp
        self.K = np.array([[-k1, 0, 0,0],
                           [0, -k2, -k3, -k4]])

        self.solve_time = 0.0
    def wrap_to_pi(self, a):
        return (a + np.pi) % (2 * np.pi) - np.pi

    def feedback_control(self, curr_state, ref_state, ref_vel_cmd):
        """
        :param curr_state: [x, y, theta, phi]
        :param ref_state: [x_d, y_d, theta_d, phi_d]
        :return: vel_cmd:[v, phi_dot]
        """
        start_time = time.time()
        v_d, phi_dot_d = ref_vel_cmd
        state_diff = ref_state - curr_state
        state_diff[2] = self.wrap_to_pi(state_diff[2])   # theta error
        state_diff[3] = self.wrap_to_pi(state_diff[3])   # phi error
        frame_rot = np.array([[np.cos(curr_state[2]), np.sin(curr_state[2]), 0, 0],
                              [-np.sin(curr_state[2]), np.cos(curr_state[2]), 0, 0],
                              [0, 0, 1, 0],
                              [0, 0, 0, 1]])
        error = frame_rot @ state_diff
        u = self.K @ error
        v = v_d * np.cos(error[2]) - u[0]
        phi_dot = phi_dot_d - u[1]
        vel_cmd = np.array([v, phi_dot])
        self.solve_time = time.time() - start_time
        return vel_cmd


