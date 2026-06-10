

# %%
# SIMULATION SETUP

import numpy as np
import matplotlib.pyplot as plt
import geometric_mpc
from GMPC_Tracking_Control_main.utils.enum_class import TrajType, ControllerType
from ref_traj_generator import TrajGenerator



# Ackermann vehicle parameters
#controller
def wrap_to_pi(a):
    return (a+np.pi)% (2*np.pi) -np.pi

controller_type = ControllerType.GMPC
init_state = np.array([0, 0, 0])
traj_config = {'type': 'eight_easy',
                   'param': {'start_state': np.array([0, 0, 0]),
                             'dt': 0.1,
                              'linear_vel': 0.1,
                              'angular_vel': 0.1,  # don't change this
                             'nTraj': 600}}

traj_gen = TrajGenerator(traj_config)
ref_state, ref_control, dt = traj_gen.get_traj()


controller = geometric_mpc.GeometricMPC(traj_config)
Q = np.array([20000, 20000, 2000])
R = 0.3
N = 10


controller.setup_solver(Q, R, N)

ref_state, ref_control, dt = traj_gen.get_traj()

L = 0.256


nTraj = ref_state.shape[1]

x = np.zeros((3, nTraj))
u = np.zeros((2, nTraj))

x[0,0] = -1
x[1,0] = -1
x[2,0] = 0
t = 0

v_min = -1.5
v_max= 1.5
w_min = -3.0
w_max = 3.0

controller.set_control_bound(v_min, v_max, w_min, w_max)


for i in range(1,nTraj):
    
    curr_state = x[:,i-1]  + dt*np.array([u[0,i-1]*np.cos(x[2,i-1]), 
                                         u[0,i-1]*np.sin(x[2,i-1]),
                                  u[0,i-1] * 1.0 / L* np.tan(u[1,i-1])])
    
    x[:,i] = curr_state
    x[2,i] = wrap_to_pi(x[2,i])

    u[:, i] = controller.solve(x[:, i], t)

    u[1, i] = np.arctan2(L * u[1, i],u[0,i])  # convert curvature to steering angle

    u[1,i] = np.clip(u[1,i],-0.5,0.5)
    
    
   
    t += dt
    print("step",i,"out of ",nTraj)
    


t = np.arange(0.0,nTraj*dt, dt)
# Plot the states as a function of time
fig1 = plt.figure(1)
fig1.set_figheight(6.4)
ax1a = plt.subplot(311)
plt.plot(t, x[0, :])
plt.plot(t, ref_state[0,:],'r--')
plt.grid(color="0.95")
plt.ylabel(r"$x$ [m]")
plt.setp(ax1a, xticklabels=[])
ax1b = plt.subplot(312)
plt.plot(t, x[1, :])
plt.plot(t, ref_state[1,:],'r--')
plt.grid(color="0.95")
plt.ylabel(r"$y$ [m]")
plt.setp(ax1b, xticklabels=[])
ax1c = plt.subplot(313)
plt.plot(t, x[2, :] * 180.0 / np.pi)
plt.plot(t, ref_state[2,:]* 180.0 / np.pi, 'r--')
plt.grid(color="0.95")
plt.ylabel(r"$\theta$ [deg]")
plt.xlabel(r"$t$ [s]")
plt.legend()
# Save the plot
#plt.savefig("../agv-book/figs/ch3/ackermann_kinematic_fig1.pdf")

# Plot the position of the vehicle in the plane
fig2 = plt.figure(2)
plt.plot(x[0, :], x[1, :])
plt.plot(ref_state[0, :], ref_state[1, :], 'b--')
plt.axis("equal")


fig3 = plt.figure(3)
ax1b = plt.subplot(211)
plt.plot(t, u[0, :])
plt.grid(color="0.95")
plt.ylabel(r"$v$ [m/s]")
plt.setp(ax1b, xticklabels=[])
ax1c = plt.subplot(212)
plt.plot(t, u[1, :] * 180.0 / np.pi)
plt.grid(color="0.95")
plt.ylabel(r"$\phi$ [deg]")
plt.xlabel(r"$t$ [s]")
plt.legend()
plt.plot()
# Save the plot
#plt.savefig("../agv-book/figs/ch3/ackermann_kinematic_fig2.pdf")

# Show all the plots to the screen
plt.show()

# %%
# MAKE AN ANIMATION

# Create and save the animation
ani = vehicle.animate(
    x,
    T,
    0,
    0,
    True,
    "../agv-book/gifs/ch3/ackermann_kinematic.gif",
)

# Show the movie to the screen
plt.show()

# # Show animation in HTML output if you are using IPython or Jupyter notebooks
# plt.rc('animation', html='jshtml')
# display(ani)
# plt.close()
