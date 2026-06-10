import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
from manifpy import SE2, SE2Tangent, SO2, SO2Tangent
import math
from autonomy.GMPC.ref_traj_generator import TrajGenerator
import casadi as ca
from autonomy.GMPC.enum_class import CostType, DynamicsType, ControllerType
from autonomy.GMPC.symbolic_system import FirstOrderModel
import time

"""
naive MPC for unicycle model
"""


class BycicleModel:
    def __init__(self, config: dict = {}, dt = 0.05, **kwargs):
        self.nState = 4
        self.nControl = 2

        self.dt = dt

        # setup configuration
        self.config = {'cost_type': CostType.POSITION_EULER, 'dynamics_type': DynamicsType.EULER_FIRST_ORDER}
        if self.config['dynamics_type'] == DynamicsType.EULER_FIRST_ORDER:
            self.set_up_euler_first_order_dynamics()

        print(config.get("dynamics_type"))

    def set_up_euler_first_order_dynamics(self):
        print("Setting up Euler first order dynamics")
        nx = self.nState
        nu = self.nControl
        # state
        x = ca.MX.sym('x')
        y = ca.MX.sym('y')
        theta = ca.MX.sym('theta')
        phi = ca.MX.sym('phi')
        X = ca.vertcat(x, y, theta, phi)
        # controlcost

        v = ca.MX.sym('v')  # linear velocity
        steering_ratio = ca.MX.sym('steering_ratio')  # angular velocity
        U = ca.vertcat(v, steering_ratio)

        # state derivative
        x_dot = ca.cos(theta) * v
        y_dot = ca.sin(theta) * v
        theta_dot = v * ca.tan(phi) / 0.256
        phi_dot = steering_ratio
        X_dot = ca.vertcat(x_dot, y_dot, theta_dot, phi_dot)

        # cost function
        self.costType = self.config['cost_type']
        print("Cost type: {}".format(self.costType))
        Q = ca.MX.sym('Q', nx, nx)
        R = ca.MX.sym('R', nu, nu)
        Xr = ca.MX.sym('Xr', nx, 1)
        Ur = ca.MX.sym('Ur', nu, 1)
        if self.costType == CostType.POSITION:
            cost_func = 0.5 * (X[:2] - Xr[:2]).T @ Q[:2, :2] @ (X[:2] - Xr[:2]) + 0.5 * (U - Ur).T @ R @ (U - Ur)
        elif self.costType == CostType.POSITION_EULER:
            pos_cost = 0.5 * (X[:2] - Xr[:2]).T @ Q[:2, :2] @ (X[:2] - Xr[:2])
            theta = X[2]
            theta_target = Xr[2]
            dth = theta - theta_target
            qth = Q[2,2]

            euler_cost = 0.5*qth*(1 - ca.cos(dth))
        
            phi = X[3]
            phi_target = Xr[3]
            dphi = phi - phi_target
            qphi = Q[3,3]

            euler_phi_cost = 0.5*qphi*(1 - ca.cos(dphi))
            cost_func = pos_cost + euler_cost + euler_phi_cost + 0.5 * (U - Ur).T @ R @ (U - Ur)

        cost = {'cost_func': cost_func, 'vars': {'X': X, 'Xr': Xr, 'U': U, 'Ur': Ur, 'Q': Q, 'R': R}}
    
        # define dynamics and cost dict
        dynamics = {'dyn_eqn': X_dot, 'vars': {'X': X, 'U': U}}
        params = {
            'X_EQ': np.zeros(self.nState),  # np.atleast_2d(self.X_GOAL)[0, :],
            'U_EQ': np.zeros(self.nControl)  # np.atleast_2d(self.U_GOAL)[0, :],
        }
        self.symbolic = FirstOrderModel(dynamics, cost, self.dt, params)
    
class NonlinearMPC:
    def __init__(self, model_config={}, dt=0.05):
        self.controllerType = ControllerType.NMPC
        config = model_config
        # dynamics
        self.model = BycicleModel(config, dt=dt).symbolic
        self.nState = self.model.nx  # 4 (x, y, theta,phi)
        self.nControl = self.model.nu  # 2 (v, steering_ratio)
        self.solve_time = 0.0
        self.setup_solver()
        self.set_control_bound()
        self.cost_func = self.model.cost_func

    def setup_solver(self, q=[200, 200, 0, 0], R=0.8, N=10):
        self.Q = np.diag(q)
        self.R = R * np.eye(self.model.nu)
        self.N = N

    def set_control_bound(self, v_min = -100, v_max= 100):
        self.v_min = v_min
        self.v_max = v_max

    def update_ref_traj(self, ref_state, ref_control, dt):
        self.ref_state = ref_state
        self.ref_control = ref_control
        self.dt = dt

    def solve(self, state):
        """
        state: [x, y, theta,phi]
        t: time -> index of reference trajectory (t = k * dt)
        """
        start_time = time.time()
        if self.ref_state is None:
            raise ValueError('Reference trajectory is not set up yet!')

        nu = self.nControl
        nx = self.nState
        N = self.N
        X = self.ref_state[:,  N]  # terminal state as goal
        x_goal = X
        opti = ca.Opti()
        x_var = opti.variable(nx, N + 1)
        u_var = opti.variable(nu, N)

        # initial state constraint
        opti.subject_to(x_var[:, 0] == state)

        # dynamics constraint
        for i in range(N):
            # Euler first order
            x_next = x_var[:, i] + self.dt * self.model.fc_func(x_var[:, i], u_var[:, i])
            opti.subject_to(x_var[:, i + 1] == x_next)

        # cost function
        cost = 0

        for i in range(N):
            x_target = self.ref_state[:, i]
            u_target = self.ref_control[:, i]
            # u_target = np.zeros((2, 1))
            cost += self.cost_func(x_var[:, i], x_target, u_var[:, i], u_target, self.Q, self.R)

        cost += self.cost_func(x_var[:, N], x_goal, np.zeros((nu,1)), np.zeros((nu, 1)), 100*self.Q, self.R)
        # control bound
        opti.subject_to(u_var[0, :] >= self.v_min)
        opti.subject_to(u_var[0, :] <= self.v_max)
        opti.subject_to(x_var[3, :] >= -0.5)
        opti.subject_to(x_var[3, :] <= 0.5)


        opti.minimize(cost)
        opts_setting = {'ipopt.max_iter': 1000, 'ipopt.print_level': 0, 'print_time': 0, 'ipopt.acceptable_tol': 1e-8,
                        'ipopt.acceptable_obj_change_tol': 1e-6}

        opti.solver('ipopt', opts_setting)
        sol = opti.solve()
        u = sol.value(u_var[:, 0])
        self.solve_time = time.time() - start_time
        return u

    def get_solve_time(self):
        return self.solve_time

    def vel_cmd_to_local_twist(self, vel_cmd):
        return ca.vertcat(vel_cmd[0], 0, vel_cmd[1])

    def local_twist_to_vel_cmd(self, local_vel):
        return ca.vertcat(local_vel[0], local_vel[2])

    @property
    def get_controller_type(self):
        return self.controllerType




