#! /usr/bin/env python3


# Generic python packages
import time  # Time library
import numpy as np
import csv
import yaml
import cvxpy as cp
from qcar2_controller.controller_config.gmpc_ackermann import GeometricMPC_ackermann as gmpc_ackermann

# ROS specific packages
from rclpy.duration import Duration # Handles time for ROS 2
import rclpy # Python client library for ROS 2
from geometry_msgs.msg import PoseStamped, Point, Quaternion, Pose,Twist, TwistStamped # Pose with ref frame and timestamp
from rclpy.node import Node
from rcl_interfaces.msg import SetParametersResult
from rclpy.qos import QoSProfile, ReliabilityPolicy
from scipy.spatial.transform import Rotation
from std_msgs.msg import Bool, Float32
from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState
from pathlib import Path

class GMPC_ackermann_node(Node):

    def __init__(self):
        super().__init__('GMPC_ackermann_node')

        # =========================================================
        # Parameters
        # =========================================================
        self.declare_parameter('qcarnumber', 1)
        self.qcarnumber = self.get_parameter('qcarnumber').get_parameter_value().integer_value

        self.declare_parameter('config_dir', '')
        self.config_dir = self.get_parameter('config_dir').get_parameter_value().string_value
        if self.config_dir == '':
            raise ValueError('config_dir parameter was not provided')
        

        # =========================================================
        # State holders / initial values
        # =========================================================
        self.position = Point()
        self.orientation = Quaternion()

        self.position.x = 0.0
        self.position.y = 0.0
        self.position.z = 0.0

        self.phi = 0.0
        self.yaw = 0.0

        self.dt = 0.05  # control period

        self.max_steering_angle = 0.6  # steering limit [rad]
        self.ell = 0.256               # wheelbase

        self.flag_info = False

        # =========================================================
        # Controller initialization
        # =========================================================
        self.controller = gmpc_ackermann()
        self.config_dir = Path(self.config_dir)

        gmpc_ackermann_config_path = self.config_dir / "gmpc_ackermann_tuning.yaml"
        with open(gmpc_ackermann_config_path, "r") as f:
            cfg = yaml.safe_load(f)

        gmpc_ackermann_cfg = cfg["gmpc_ackermann"]
        key = f"qcar{self.qcarnumber}"

        if key in gmpc_ackermann_cfg:
            params = gmpc_ackermann_cfg[key]
        else:
            self.get_logger().error(f"Configuration for {key} not found in GMPC config file.")
            raise KeyError(f"Configuration for {key} not found.")

        self.Q = np.array(params["Q"], dtype=float)
        self.R = np.array(params["R"], dtype=float)
        self.N = int(params["N"])

        # Allocate buffers dependent on N
        self.desired_state = np.zeros((4, self.N + 1))
        self.desired_control = np.zeros((3, self.N + 1))

        # Setup solver + bounds
        self.controller.setup_solver(self.Q, self.R, self.N)
        v_min = -1.75
        v_max = 1.75
        phi_min = -0.5
        phi_max = 0.5
        self.controller.set_control_bound(v_min, v_max, phi_min, phi_max)

        # =========================================================
        # Error simulation configuration
        # =========================================================
        error_simulation_path = self.config_dir / "error_simulation_config.yaml"
        with open(error_simulation_path, "r") as f:
            error_params = yaml.safe_load(f)

        self.simulate_errors_enable = bool(error_params["simulate_errors_enable"])
        self.simulation_error_type = error_params["error_type"]
        self.error_magnitude = float(error_params["error"])
        self.gps_sigma = float(error_params["gps_sigma"])
        self.yaw_sigma = np.deg2rad(float(error_params["yaw_sigma"]))
        self.phi_sigma = np.deg2rad(float(error_params["phi_sigma"]))
        self.seed = int(error_params["seed"])

        np.random.seed(self.seed)

        # =========================================================
        # FSM / commands
        # =========================================================
        self.desired_steering_angle = 0.0       
        self.current_steering_angle = 0.0 
        self.steering_time_constant = 0.16  # time constant for first order filter on steering angle feedback (if used)
        self.FSM = 0

        # =========================================================
        # Publisher(s)
        # =========================================================
        self.publisher = self.create_publisher(Twist, 'cmd_vel_nav', 1)
        self.solve_time_publisher = self.create_publisher(Float32, 'controller_solve_time', 1)

        # =========================================================
        # Subscriptions (main loop + inputs)
        # =========================================================
        self.path_control_subscription = self.create_subscription(
            Odometry,
            'tracking_waypoint',
            self.control_algorithm,
            20
        )  # main loop

        self.subscription_vycon = self.create_subscription(
            PoseStamped,
            'vrpn_pose',
            self.pose_vycon_callback,
            QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, depth=10)
        )

        # self.subscription_follower_velocity = self.create_subscription(
        #     TwistStamped,
        #     '/qcar2_1/vrpn_mocap/Qcar2_1/twist',
        #     self.follower_velocity_callback,
        #     QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, depth=10)
        # )

        self.subscription_stop_flag = self.create_subscription(
            Bool,
            '/qcar/stop',
            self.stop_experiment_callback,
            10
        )

        self.subscription_update_waypoints = self.create_subscription(
            Odometry,
            'desired_trajectory',
            self.update_waypoints_callback,
            20
        )

        self.path_following_enable_sub = self.create_subscription(
            Bool,
            'path_following_enable',
            self.path_following_enable_callback,
            10
        )

        self.get_logger().info("Ready to run GMPC version PHI DOT controller")
    
    def update_waypoints_callback(self, msg):
      # =========================================================
      # 1) Read pose (world)
      # =========================================================
      x = msg.pose.pose.position.x
      y = msg.pose.pose.position.y
      q = msg.pose.pose.orientation
      _, _, yaw = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_euler('xyz', degrees=False)
      phi = msg.twist.twist.angular.y  # Retrieve phi from angular.y for now

      # =========================================================
      # 2) Read twist (world)
      # =========================================================
      v = msg.twist.twist.linear.x
     
      omega = msg.twist.twist.angular.z
      phi_dot = msg.twist.twist.angular.x  # Retrieve phi_dot from angular.x


      # =========================================================
      # 4) Optional noise injection
      # =========================================================
      if self.simulate_errors_enable:
          x, y, yaw, phi, v, omega, phi_dot = self.apply_simulated_noise(x, y, yaw, phi, v, omega, phi_dot)

      # =========================================================
      # 6) Build new desired state/control sample
      # =========================================================
      desired_state_received = np.array([x, y, yaw, phi])


      desired_control_received = np.array([v, omega, phi_dot])

      # =========================================================
      # 7) Shift buffers and append newest sample
      # =========================================================
      self.desired_state[:, :-1] = self.desired_state[:, 1:]
      self.desired_control[:, :-1] = self.desired_control[:, 1:]

      self.desired_state[:, -1] = desired_state_received
      self.desired_control[:, -1] = desired_control_received


      # =========================================================
      # 8) Update controller
      # =========================================================

      self.controller.update_ref_traj(self.desired_state, self.desired_control, self.dt)
      
    def apply_simulated_noise(self, x, y, yaw,phi, v, omega,phi_dot):
      """
      Applies simulated noise to states depending on self.simulation_error_type.
      Returns possibly modified (x, y, yaw, v, omega).
      """

      if self.simulation_error_type == "speed_noise":
          v += self.error_magnitude * v * np.random.normal(0, 1)

      elif self.simulation_error_type == "omega_noise":
          omega += self.error_magnitude * omega * np.random.normal(0, 1)

      elif self.simulation_error_type == "speed_omega_noise":
          v += self.error_magnitude * v * np.random.normal(0, 1)
          omega += self.error_magnitude * omega * np.random.normal(0, 1)

      elif self.simulation_error_type == "all_states":
          v += self.error_magnitude * v * np.random.normal(0, 1)
          omega += self.error_magnitude * omega * np.random.normal(0, 1)
          phi_dot += self.error_magnitude * phi_dot * np.random.normal(0, 1)
          x += self.gps_sigma * np.random.normal(0, 1)
          y += self.gps_sigma * np.random.normal(0, 1)
          phi += self.phi_sigma * np.random.normal(0, 1)
          yaw += self.yaw_sigma * np.random.normal(0, 1)

      return x, y, yaw, phi, v, omega, phi_dot
    
    
    
    def path_following_enable_callback(self, msg: Bool):
      flag_enable_following_path = msg.data
      if flag_enable_following_path:
        if not self.flag_info:
          self.get_logger().info("Path following Initiated")
        self.flag_info = True
      else:
        if self.flag_info:
          self.get_logger().info("Path enable disabled by planner")
        self.flag_info = False
        self.nav_command(0.0, self.desired_steering_angle)


    # mocap pose callback
    def pose_vycon_callback(self,msg):
      self.position = msg.pose.position
      orientation = msg.pose.orientation  
      rotation = [orientation.x, orientation.y, orientation.z, orientation.w]
      roll, pitch, self.yaw = Rotation.from_quat(rotation).as_euler('xyz', degrees=False)
    

    # def follower_velocity_callback(self, msg):
    #   vx = msg.twist.linear.x
    #   vy = msg.twist.linear.y
    #   omega = msg.twist.angular.z
    #   v_mag = np.hypot(vx, vy)

    #   v_sign = np.sign(vx * np.cos(self.yaw) + vy * np.sin(self.yaw))

    #   v = v_mag * v_sign

    #   self.phi = np.arctan(omega * self.ell / v) if abs(v) > 0.05 else self.steering_angle  # if speed is very low, keep previous steering angle to avoid singularity

    def stop_experiment_callback(self, msg: Bool):
      self.FSM = msg.data
      if not self.FSM:
        self.get_logger().info("User called STOP ")
        self.nav_command(0.0,self.desired_steering_angle)
      else:
        self.get_logger().info("User called START ")
    

    def control_algorithm(self, msg):

      # current position
      time_start = time.time()
      x = np.array([self.position.x, self.position.y, self.yaw, self.current_steering_angle])

      if self.FSM == 1:

          commands_star = self.controller.solve(x)
          # self.get_logger().info('GMPC working')

          speed_command = commands_star[0]
          phi_dot = commands_star[1]

          self.desired_steering_angle = self.current_steering_angle + phi_dot * self.dt

          self.desired_steering_angle = np.clip(
              self.desired_steering_angle,
              -self.max_steering_angle,
              self.max_steering_angle
          )

      else:
          speed_command = 0.0

      if x[0] > 3.4 or x[0] < -3.4 or x[1] > 2 or x[1] < -2:
          speed_command = 0.0


      # self.get_logger().info(f"GMPC computation time: {time_end - time_start} seconds")
      time_end = time.time()
      elapsed_time = time_end - time_start
      self.nav_command(speed_command, self.desired_steering_angle)
      self.calculate_current_steering()
      self.publish_solve_time(elapsed_time)
    
    def publish_solve_time(self, solve_time):
      solve_time_msg = Float32()
      solve_time_msg.data = solve_time
      self.solve_time_publisher.publish(solve_time_msg)
           

    def nav_command(self,speed_command, steering_angle):
      QCarCommands = Twist()
      QCarCommands.linear.x = speed_command
      QCarCommands.angular.z = steering_angle
      self.publisher.publish(QCarCommands)

    def calculate_current_steering(self):
       a_phi = 1.0 - np.exp(-self.dt / self.steering_time_constant)
       self.current_steering_angle = self.current_steering_angle + a_phi * (self.desired_steering_angle - self.current_steering_angle)
       
    
      

       


def main():


  # Start the ROS 2 Python Client Library
  rclpy.init()

  node = GMPC_ackermann_node()
  try:
      rclpy.spin(node)
  except KeyboardInterrupt:
      speed_command = 0.0
      steering_angle = 0.0
      node.nav_command(speed_command,steering_angle)
      
  node.destroy_node()
  rclpy.shutdown()

if __name__ == '__main__':
  main()