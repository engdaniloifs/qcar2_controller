#! /usr/bin/env python3


# Generic python packages
import time
import numpy as np
import yaml


# ROS specific packages
from rclpy.duration import Duration # Handles time for ROS 2
import rclpy # Python client library for ROS 2
# from geometry_msgs.msg import PoseStamped, Point, Quaternion, Pose,Twist # Pose with ref frame and timestamp
from rclpy.node import Node
# from rclpy.qos import QoSProfile, ReliabilityPolicy
# from scipy.spatial.transform import Rotation as Rotation_test
from std_msgs.msg import Bool, Float32
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, TwistStamped, Quaternion, Pose,Twist # Pose with ref frame and timestamp
from scipy.spatial.transform import Rotation
from rclpy.qos import QoSProfile, ReliabilityPolicy




class Path_planner(Node):

    def __init__(self):
      super().__init__('path_planner_node')
      # =========================================================
      # Parameters
      # =========================================================
      self.declare_parameter('qcarnumber', 1)
      self.qcarnumber = self.get_parameter('qcarnumber') \
          .get_parameter_value().integer_value

      self.declare_parameter('controller_type', 'NMPC')
      self.controller_type = self.get_parameter('controller_type') \
          .get_parameter_value().string_value

      # =========================================================
      # Timing
      # =========================================================
      self.dt = 0.05

      # =========================================================
      # Flags
      # =========================================================
      self.fill_horizon_flag = True
      self.initialize_plot_error = False

      # =========================================================
      # State Holders
      # =========================================================
      self.last_leader_state = np.array([0.0, 0.0, 0.0])
      self.last_leader_velocity = np.array([0.0, 0.0, 0.0])
      self.last_follower_state = np.array([0.0, 0.0, 0.0])

      self.offset_follower_too_close = 0.42 + 0.3  # meters behind on local x-axis

      # =========================================================
      # Controller Configuration Selection
      # =========================================================
      if self.controller_type == "FBLinear":
          config_path = r"/home/nvidia/Documents/parameters_leader_follower/fblinearization_tuning.yaml"
          controller_name = "fblinearization"

      elif self.controller_type == "NMPC":
          config_path = r"/home/nvidia/Documents/parameters_leader_follower/nmpc_tuning.yaml"
          controller_name = "nmpc"

      elif self.controller_type == "GMPC":
          config_path = r"/home/nvidia/Documents/parameters_leader_follower/gmpc_tuning.yaml"
          controller_name = "gmpc"

      elif self.controller_type == "GMPC_ackermann":
          config_path = r"/home/nvidia/Documents/parameters_leader_follower/gmpc_ackermann_tuning.yaml"
          controller_name = "gmpc_ackermann"
      elif self.controller_type == "GMPC_phi":
          config_path = r"/home/nvidia/Documents/parameters_leader_follower/gmpc_phi_tuning.yaml"
          controller_name = "gmpc_phi"

      # =========================================================
      # Load YAML Parameters
      # =========================================================
      with open(config_path, "r") as f:
          cfg = yaml.safe_load(f)
          controller_cfg = cfg[controller_name]
          key = f"qcar{self.qcarnumber}"

      if key in controller_cfg:
          params = controller_cfg[key]
      else:
          self.get_logger().error(
              f"Configuration for {key} not found in {controller_name} config file."
          )
          raise KeyError(f"Configuration for {key} not found.")

      self.Q = np.array(params["Q"], dtype=float) if "Q" in params else np.nan
      self.R = np.array(params["R"], dtype=float) if "R" in params else np.nan
      self.N = int(params["N"])
      if self.controller_type in ["NMPC","GMPC_ackermann"]:
         self.N += 1  # Account for the extra state/control at the end of the horizon

      # =========================================================
      # Timers
      # =========================================================
      self.path_control_timer = self.create_timer(self.dt, self.planner)

      # =========================================================
      # Publishers
      # =========================================================
      self.publisher_trajectory = self.create_publisher(
          Odometry, 'desired_trajectory', 20
      )

      self.publisher_tracking_waypoint = self.create_publisher(
          Odometry, 'tracking_waypoint', 20
      )

      self.path_enable_publisher = self.create_publisher(
          Bool, 'path_following_enable', 10
      )

      self.publisher_error_plot_trajectory = self.create_publisher(
          Odometry, 'error_plot_trajectory', 1
      )

      # =========================================================
      # FSM
      # =========================================================
      self.FSM = 0  # (0: stop, 1: go)

      # =========================================================
      # Reference Buffers
      # =========================================================
      self.leaderbuffer = []

      # =========================================================
      # Subscriptions
      # =========================================================
      self.subscription_follower_pose = self.create_subscription(
          PoseStamped,
          '/qcar2_1/vrpn_mocap/Qcar2_1/pose',
          self.follower_pose_callback,
          QoSProfile(
                                    reliability=ReliabilityPolicy.BEST_EFFORT,
                                    depth=10
                                )
      )

      self.subscription_leader_pose = self.create_subscription(
          PoseStamped,
          '/qcar2_1/vrpn_mocap/Qcar2_2/pose',
          self.leader_pose_callback,
          QoSProfile(
                                    reliability=ReliabilityPolicy.BEST_EFFORT,
                                    depth=10
                                )
      )

      self.subscription_leader_velocity = self.create_subscription(
          TwistStamped,
          '/qcar2_1/vrpn_mocap/Qcar2_2/twist',
          self.leader_velocity_callback,
          QoSProfile(
                                    reliability=ReliabilityPolicy.BEST_EFFORT,
                                    depth=10
                                )
      )

      self.subscription_follower_velocity = self.create_subscription(
          TwistStamped,
          '/qcar2_1/vrpn_mocap/Qcar2_1/twist',
          self.follower_velocity_callback,
          QoSProfile(
                                    reliability=ReliabilityPolicy.BEST_EFFORT,
                                    depth=10
                                )
      )

      self.subscription_stop_flag = self.create_subscription(
          Bool,
          '/qcar/stop',
          self.stop_experiment_callback,
          10
      )
      self.subscription_steering_angle = self.create_subscription(
          Float32,
          '/qcar2_2/current_steering_angle',
          self.leader_steering_angle_callback,
            10
      )

      self.get_logger().info("Path planner node has been started.")
      
    
    def stop_experiment_callback(self, msg: Bool):
      self.FSM = msg.data
      if not self.FSM:
        self.get_logger().info("User called STOP ")
      else:
        self.get_logger().info("User called START")

    def leader_steering_angle_callback(self, msg: Float32):
      self.leader_steering = msg.data

    def leader_pose_callback(self, msg):
      x = msg.pose.position.x
      y = msg.pose.position.y
      q = msg.pose.orientation
      roll, pitch, yaw = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_euler('xyz', degrees=False)

      # Store them as arrays
      self.last_leader_state = np.array([x, y, yaw])

    def follower_pose_callback(self, msg):
      x = msg.pose.position.x
      y = msg.pose.position.y
      q = msg.pose.orientation
      roll, pitch, yaw = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_euler('xyz', degrees=False)

      # Store them as arrays
      self.last_follower_state = np.array([x, y, yaw])

    def follower_velocity_callback(self, msg):
      vx = msg.twist.linear.x
      vy = msg.twist.linear.y
      omega = msg.twist.angular.z
      # Store them as arrays
      self.last_follower_velocity = np.array([vx, vy, omega])
      
    def leader_velocity_callback(self, msg):
      vx = msg.twist.linear.x
      vy = msg.twist.linear.y
      omega = msg.twist.angular.z
      # Store them as arrays
      self.last_leader_velocity = np.array([vx, vy, omega])

    def distance(self, state1, state2):
      return np.sqrt((state1[0] - state2[0])**2 + (state1[1] - state2[1])**2)
    
    def buffer_distance(self):

        # =========================================================
        # 1) Append latest leader sample to buffer
        # =========================================================
        ref_state = self.last_leader_state
        vx,vy,omega = self.last_leader_velocity
        v_mag = np.hypot(vx, vy)
        v_sign = np.sign(vx * np.cos(ref_state[2]) + vy * np.sin(ref_state[2]))
        v = v_mag * v_sign
        phi = self.leader_steering
        _,_,_,prev_phi = self.leaderbuffer[-1][0] if len(self.leaderbuffer) > 0 else (0,0,0,phi)
        phi_dot = (phi - prev_phi) / self.dt
        ref_state = np.array([ref_state[0], ref_state[1], ref_state[2], phi])
        ref_control = np.array([v,omega, phi_dot])
        self.leaderbuffer.append((ref_state, ref_control))

        # =========================================================
        # 2) Extract up to N samples for MPC horizon
        # =========================================================
        H = min(self.N, len(self.leaderbuffer))

        leader_state = [self.leaderbuffer[i][0] for i in range(H)]
        leader_control = [self.leaderbuffer[i][1] for i in range(H)]

        # =========================================================
        # 3) Check if buffer has reached required time length
        # =========================================================
        if len(self.leaderbuffer) > 70:
            buffer_time = True
        else:
            self.get_logger().info("filling buffer")
            buffer_time = False

        return leader_state, leader_control, buffer_time

    def planner(self):
        
        if self.FSM == 1:
            # flag3 indicates the controller has started; after one timestep we can compute and publish error
            if self.initialize_plot_error:
                plot_leader_state, plot_leader_control = self.leaderbuffer[0][0], self.leaderbuffer[0][1]
                self.calculate_trajectory_error(
                    plot_leader_state,
                    plot_leader_control,
                    self.last_follower_state,
                    self.last_follower_velocity
                )
                self.leaderbuffer.pop(0)

            leader_state, leader_control, buffer_time_flag = self.buffer_distance()

            # after the buffer has filled the mpc horizon, we can fill the mpc buffer
            if self.fill_horizon_flag and len(leader_state) == self.N:
                for i in range(0, self.N):
                    self.publish_desired_trajectory(leader_state[i], leader_control[i])

                self.get_logger().info("filling mpc buffer done")
                self.fill_horizon_flag = False


            if buffer_time_flag:

                if self.distance(self.last_follower_state, self.last_leader_state) < self.offset_follower_too_close:
                    self.FSM = 2
                    self.path_following_enable_publisher(False)
                    self.get_logger().info("Follower too close to Leader, stopping path following")
                else:
                    self.path_following_enable_publisher(True)
                    self.initiate_GMPC_path_following(leader_state[0], leader_control[0])

                    following_state, following_control = leader_state[-1], leader_control[-1]

                    self.publish_desired_trajectory(following_state, following_control)
                    self.initialize_plot_error = True
        if self.FSM == 2:
            leader_state, leader_control, buffer_time = self.buffer_distance()

            if self.distance(self.last_follower_state, self.last_leader_state) > self.offset_follower_too_close:
                self.FSM = 1
        if self.FSM == 0:
            self.path_following_enable_publisher(False)
        

    def publish_desired_trajectory(self,desired_state, desired_control):
        msg = Odometry()

        x, y, yaw, phi = desired_state
        v, omega,phi_dot = desired_control

        msg = Odometry()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "world"     # Reference frame
        msg.child_frame_id = "leader"     # Robot frame

        # Pose
        msg.pose.pose.position.x = float(x)
        msg.pose.pose.position.y = float(y)
        msg.pose.pose.position.z = 0.0

        # Convert yaw -> quaternion
        q = Rotation.from_euler('xyz', [0.0, 0.0, yaw]).as_quat()
        msg.pose.pose.orientation.x = q[0]
        msg.pose.pose.orientation.y = q[1]
        msg.pose.pose.orientation.z = q[2]
        msg.pose.pose.orientation.w = q[3]

        # Twist
        msg.twist.twist.linear.x = float(v)
        msg.twist.twist.linear.y = 0.0 
        msg.twist.twist.angular.z = float(omega)
        msg.twist.twist.angular.x = float(phi_dot)  # Store phi_dot in angular.x for now
        msg.twist.twist.angular.y = float(phi)      # Store phi in angular.y for now

        self.publisher_trajectory.publish(msg)

    def initiate_GMPC_path_following(self,desired_state, desired_control):
        msg = Odometry()

        x, y, yaw,phi = desired_state
        vx, vy, omega = desired_control

        msg = Odometry()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "world"     # Reference frame
        msg.child_frame_id = "leader"     # Robot frame

        # Pose
        msg.pose.pose.position.x = float(x)
        msg.pose.pose.position.y = float(y)
        msg.pose.pose.position.z = 0.0

        # Convert yaw -> quaternion
        q = Rotation.from_euler('xyz', [0.0, 0.0, yaw]).as_quat()
        msg.pose.pose.orientation.x = q[0]
        msg.pose.pose.orientation.y = q[1]
        msg.pose.pose.orientation.z = q[2]
        msg.pose.pose.orientation.w = q[3]

        # Twist
        msg.twist.twist.linear.x = float(vx)
        msg.twist.twist.linear.y = float(vy)  
        msg.twist.twist.angular.z = float(omega)

        self.publisher_tracking_waypoint.publish(msg)
              

    def calculate_trajectory_error(self, desired_state, desired_control, follower_state, follower_control):
      # =========================================================
      # Unpack inputs
      # =========================================================
      x_d, y_d, yaw_d, phi_d = desired_state
      v_d, omega_d, phi_dot_d = desired_control

      x_f, y_f, yaw_f = follower_state
      vx_f, vy_f, omega_f = follower_control

      # =========================================================
      # Position errors
      # =========================================================
      error_x = x_d - x_f
      error_y = y_d - y_f
      error_euclidean = np.sqrt(error_x**2 + error_y**2)

      # Wrap yaw error to [-pi, pi]
      error_yaw = (yaw_d - yaw_f + np.pi) % (2*np.pi) - np.pi
      error_yaw_deg = error_yaw * 180.0 / np.pi

      # =========================================================
      # Velocity errors
      # =========================================================
      v_follower = np.hypot(vx_f, vy_f)
      error_v = v_d - v_follower

      error_omega = omega_d - omega_f

      # Follower speed along its yaw (projection)
      absolut_v = vx_f*np.cos(yaw_f) + vy_f*np.sin(yaw_f)

      # =========================================================
      # Pack into Odometry message for plotting
      # =========================================================
      msg = Odometry()
      msg.header.stamp = self.get_clock().now().to_msg()
      msg.header.frame_id = "world"
      msg.child_frame_id = "error"

      # Pose: x/y store signed errors, z stores Euclidean norm
      msg.pose.pose.position.x = float(error_x)
      msg.pose.pose.position.y = float(error_y)
      msg.pose.pose.position.z = float(error_euclidean)

      # Orientation: currently fixed identity quaternion (not using yaw error here)
      msg.pose.pose.orientation.x = 0.0
      msg.pose.pose.orientation.y = 0.0
      msg.pose.pose.orientation.z = 0.0
      msg.pose.pose.orientation.w = 1.0

      # Twist: store velocity error components + scalar norms and angles
     
      msg.twist.twist.linear.z = float(error_v)

      msg.twist.twist.angular.x = float(error_yaw_deg)
      msg.twist.twist.angular.z = float(error_omega)

      # Abuse covariance[0] to store follower absolute speed
      msg.twist.covariance[0] = float(absolut_v)

      self.publisher_error_plot_trajectory.publish(msg)
       
    
    def path_following_enable_publisher(self, enable):
        msg = Bool()
        msg.data = enable
        self.path_enable_publisher.publish(msg)




def main():

  # Start the ROS 2 Python Client Library
  rclpy.init()

  node = Path_planner()
  try:
      rclpy.spin(node)
  except KeyboardInterrupt:
        pass
      
  node.destroy_node()
  rclpy.shutdown()

if __name__ == '__main__':
  main()