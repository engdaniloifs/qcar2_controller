import numpy as np
import scipy.linalg
import time
import matplotlib.pyplot as plt
from manifpy import SE2, SE2Tangent, SO2, SO2Tangent
import casadi as ca
import math
from autonomy.GMPC.ref_traj_generator import TrajGenerator
from autonomy.GMPC.enum_class import TrajType, ControllerType, LiniearizationType

"""
this GeometricMPC class is used to solve tracking problem of uni-cycle model
using MPC. The error dynamics is defined as follows:
error dynamics:
    psi_dot = At * psi_t + Bt * ut + ht
state:
    psi: lie algebra element of Psi (SE2 error)
control:
    ut = xi_t: twist (se2 element)

State transition matrix:
    At: ad_{xi_d,t}
Control matrix:
    B_k = I
offset:
    ht = xi_t,d: desired twist (se2 element)
    
the reference trajectory is generated using TrajGenerator class in ref_traj_generator.py
"""


class GeometricMPC_ackermann:
    def __init__(self):
        self.nState = 4  # twist error (se2 vee) R^3
        self.nControl = 2  # velocity control (v, w) R^2
        self.nTwist = 4  # twist (se2 vee) R^3
        self.nTraj = None
        self.ref_state = None
        self.ref_control = None
        self.dt = None
        self.Q = None
        self.R = None
        self.N = None
        self.solve_time = 0.0
        self.setup_solver()
        self.set_control_bound()


    def setup_solver(self, Q=[20000, 20000, 2000], R=0.3, N=10):
        self.Q = np.diag(Q)
        self.R = R * np.diag(np.ones(self.nControl))
        self.N = N

    def set_control_bound(self, v_min = -1.75, v_max= 1.75, phi_min = -0.5, phi_max= 0.5):
        self.v_min = v_min
        self.v_max = v_max
        self.phi_min = phi_min
        self.phi_max = phi_max

    def update_ref_traj(self, ref_state, ref_control, dt):
        self.ref_state = ref_state
        self.ref_control = ref_control
        self.dt = dt



    def solve(self, current_state):
        """
        current_state: current state of the system (x, y, theta)
        t: time -> index of reference trajectory (t = k * dt)

        return:
            u: control input (v, w)
        """

        start_time = time.time()
        if self.ref_state is None:
            raise ValueError('Reference trajectory is not set up yet!')


        curr_ref = self.ref_state[:, 0]
        # get x init by calculating log between current state and reference state
        SE_error = SE2(curr_ref[0], curr_ref[1], curr_ref[2]).between(SE2(current_state[0], current_state[1], current_state[2])).log().coeffs()
        phi = current_state[3] 
        x_init = ca.vertcat(SE_error, phi)
        Q = self.Q
        R = self.R
        N = self.N
        dt = self.dt


        # setup casadi solver
        opti = ca.Opti('conic')
        # opti = ca.Opti()
        x_var = opti.variable(self.nState, N + 1)
        u_var = opti.variable(2, N)

        # setup initial condition
        opti.subject_to(x_var[:, 0] == x_init)
    
        
        # setup dynamics constraints
        # x_next = A * x + B * u + h
        for i in range(N):
            u_d = self.ref_control[:, i]  # desir
            x_ref, y_ref, yaw_ref, phi_ref = self.ref_state[:, i]
            u_twist = self.vel_cmd_to_local_twist(u_d)
            A = -SE2Tangent(u_twist).smallAdj()
            A_aug = ca.blockcat([[A, ca.DM.zeros(3, 1)], [ca.DM.zeros(1, 4)]])
            B = np.eye(self.nTwist)
            h = -self.ackermann_to_twist4(u_d)
            twist_inputs = self.ackerman_input_to_local_twist(u_var[:, i], u_d, x_var[3,i], phi_ref)
            x_next = x_var[:, i] + dt * (A_aug @ x_var[:, i] + 
                                         B @ twist_inputs + h)
            opti.subject_to(x_var[:, i + 1] == x_next)

        # cost function
        cost = 0
        for i in range(N):
            u_d = self.ref_control[:, i]
            _, _, _, phi_ref = self.ref_state[:, i]
            x_cost = ca.vertcat(
                                x_var[0, i],
                                x_var[1, i],
                                x_var[2, i],
                                x_var[3, i] - phi_ref
                            )
            u_d = self.vel_cmd_to_ackerman_input(u_d)   
            cost += ca.mtimes([x_cost.T, Q, x_cost])
            cost += ca.mtimes([(u_var[:, i]-u_d).T, R, (u_var[:, i]-u_d)])
        _,_, _, phi_ref = self.ref_state[:, N]

        x_cost_terminal = ca.vertcat(
                                x_var[0, N],
                                x_var[1, N],
                                x_var[2, N],
                                x_var[3, N] - phi_ref
                            )
        cost += ca.mtimes([x_cost_terminal.T, 100*Q, x_cost_terminal])

        # control bound
        opti.subject_to(u_var[0, :] >= self.v_min)
        opti.subject_to(u_var[0, :] <= self.v_max)
        opti.subject_to(x_var[3, :] >= self.phi_min)
        opti.subject_to(x_var[3, :] <= self.phi_max)

        opts_setting = { 'printLevel': 'none'}
        opti.minimize(cost)
        opti.solver('qpoases',opts_setting)
        sol = opti.solve()
        psi_sol = sol.value(x_var)
        u_sol = sol.value(u_var)
        end_time = time.time()
        self.solve_time = end_time - start_time
        return u_sol[:, 0]

    def get_solve_time(self):
        return self.solve_time
    
    def ackerman_input_to_local_twist(self, ackermann_input,desired_vel_cmd_input,phi,phi_ref):
        v = ackermann_input[0]
        phi_dot = ackermann_input[1]
        v_d, w_d = desired_vel_cmd_input[0], desired_vel_cmd_input[1]
        phi_d = phi_ref
        w = (np.tan(phi_d)/0.256) * (v - v_d) + (v_d/(0.256*np.cos(phi_d)**2))*(phi - phi_d)
        
        return ca.vertcat(v, 0,w, phi_dot)
    
    def ackermann_to_twist4(self, ackermann_input):
        v = ackermann_input[0]
        omega = ackermann_input[1]
        phi_dot = ackermann_input[2]
        return ca.vertcat(v, 0, 0, 0)
    
    def vel_cmd_to_ackerman_input(self, vel_cmd):
        v, phi_dot = vel_cmd[0], vel_cmd[2]
        
        
        return ca.vertcat(v, phi_dot)

    def vel_cmd_to_local_twist(self, vel_cmd): 
        return ca.vertcat(vel_cmd[0], 0, vel_cmd[1])


    @property
    def get_controller_type(self):
        return self.controllerType


def test_mpc():
    traj_config = {'type': TrajType.CIRCLE,
                   'param': {'start_state': np.array([1, 1, np.pi / 2]),
                             'linear_vel': 0.5,
                             'angular_vel': 0.5,
                             'nTraj': 170,
                             'dt': 0.05}}

    mpc = GeometricMPC(traj_config)
    ref_SE2 = mpc.ref_state
    init_state = np.array([0.5, 0.5, 0])
    t = 0
    # contrainer to store state
    state_store = np.zeros((3, mpc.nTraj))
    state_store[:, 0] = init_state
    vel_cmd_store = np.zeros((2, mpc.nTraj))
    # start simulation
    for i in range(mpc.nTraj - 1):
        state = state_store[:, i]

        vel_cmd = mpc.solve(state, t)
        vel_cmd_store[:, i] = vel_cmd
        xi = mpc.vel_cmd_to_local_twist(vel_cmd)
        X = SE2(state[0], state[1], state[2])  # SE2 state
        X = X + SE2Tangent(xi * mpc.dt)
        state_store[:, i + 1] = np.array([X.x(), X.y(), X.angle()])

        t += mpc.dt

    # plot
    plt.figure()
    plt.plot(ref_SE2[0, :], ref_SE2[1, :], 'r')
    plt.plot(state_store[0, :], state_store[1, :], 'b')
    plt.legend(['reference', 'trajectory'])

    plt.show()

    # plot distance difference
    distance_store = np.linalg.norm(state_store[0:2, :] - ref_SE2[0:2, :], axis=0)
    plt.figure()
    plt.plot(distance_store)
    plt.title('distance difference')
    plt.show()

    # plot orientation difference
    orientation_store = np.zeros(mpc.nTraj)
    for i in range(mpc.nTraj):
        X_d = SE2(ref_SE2[0, i], ref_SE2[1, i], ref_SE2[2, i])
        X = SE2(state_store[0, i], state_store[1, i], state_store[2, i])
        X_d_inv_X = SO2(X_d.angle()).between(SO2(X.angle()))
        orientation_store[i] = scipy.linalg.norm(X_d_inv_X.log().coeffs())

    plt.figure()
    plt.plot(orientation_store[0:])
    plt.title('orientation difference')

    plt.show()

    # plot velocity command
    plt.figure()
    plt.plot(vel_cmd_store[0, :-1], 'r')
    plt.plot(vel_cmd_store[1, :-1], 'b')
    plt.legend(['linear', 'angular'])
    plt.title('velocity command')
    plt.show()



if __name__ == '__main__':
    # test_generate_ref_traj()
    test_mpc()
