#! /usr/bin/env python3


# Generic python packages
import time  # Time library
import numpy as np
import csv
import os

# ROS specific packages
from rclpy.duration import Duration # Handles time for ROS 2
import rclpy # Python client library for ROS 2
from geometry_msgs.msg import PoseStamped, Point, Quaternion, Pose,Twist # Pose with ref frame and timestamp
from rclpy.node import Node
from rcl_interfaces.msg import SetParametersResult
from std_msgs.msg import String, Bool, Float32
from rclpy.qos import QoSProfile, ReliabilityPolicy
from scipy.spatial.transform import Rotation as Rotation_test
import yaml


class LeaderTrajectory(Node):

    def __init__(self):
      super().__init__('leader_trajectory')

       # --- Parameters ---
      self.declare_parameter('qcarnumber', 1)
      self.qcarnumber = self.get_parameter('qcarnumber').get_parameter_value().integer_value


       # --- State holders (pose/orientation) ---
      self.position = Point() # latest position
      self.orientation = Quaternion() # latest orientation

      # initialize pose/orientation
      self.position.x = 0.0
      self.position.y = 0.0
      self.position.z = 0.0

      self.orientation.x = 0.0
      self.orientation.y = 0.0
      self.orientation.z = 0.0
      self.orientation.w = 1.0
      

      self.dt = 0.020 # control period

      self.desired_steering =0 # steering command
      self.current_steering = 0 # steering feedback (if available)
      self.steering_constant = 0.16 # steering time constant for first order filter

      self.path_control_timer = self.create_timer(self.dt, self.follow_trajectory) # main loop

      self.start_robotx = None


      self.publisher = self.create_publisher(Twist,'cmd_vel_nav', 1) # cmd pub
      self.publisher_steering = self.create_publisher(Float32,'current_steering_angle', 10) # steering pub (if needed)
      self.max_steering_angle = 0.6 # steering limit [rad]

      self.start = 0 # task mode received from FlagsRobotsArray
      self.FSM = 0
      self.K =0
      self.L = 0.256 # wheelbase

      config_path = r"/home/nvidia/Documents/parameters_leader_follower/traj_config.yaml"

      with open(config_path, "r") as f:
          cfg = yaml.safe_load(f)

      self.traj_config = cfg["traj_config"]

      self.desired_speed = self.traj_config["param"]["linear_vel"]  # m/s
      self.radius = self.traj_config["param"]["radius"]  # m


      # --- Subscriptions ---
      # mocap pose (remapped in launch)
      self.subscription_vycon = self.create_subscription(
                                                          PoseStamped,
                                                          'vrpn_pose',  # relative, neutral name
                                                          self.pose_vycon_callback,
                                                          QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, depth=10)
                                                        )
      self.subscription_stop_flag = self.create_subscription(Bool, '/qcar/stop',self.stop_experiment_callback , 10)
      self.get_logger().info("Leader trajectory node has been started.")

    # mocap pose callback
    def pose_vycon_callback(self,msg):
      self.position = msg.pose.position
      self.orientation = msg.pose.orientation

    def stop_experiment_callback(self, msg: Bool):
      self.start = msg.data
      if not self.start:
        self.get_logger().info("User called STOP ")
      else:
        self.get_logger().info("User called START")

    def follow_trajectory(self):
      if self.start == 1:
        enable = 1
        speed_command = float(self.desired_speed)

        if self.FSM == 0:
          self.robotx, self.roboty = 0, self.position.y
          self.start_robotx = self.position.x
          self.wp_1 = np.array([self.robotx,self.roboty])
          self.get_logger().info("Starting trajectory, going to middle point")
          self.FSM = 1
        
        if self.FSM == 1:

            # --- target extraction ---
            
            #wp_1_mod = [0,0]


            wp_1_mod = self.wp_1

            self.L= 0.256 # wheelbase

            # --- orientation (yaw) from quaternion ---
            q = self.orientation
            rotation = [q.x, q.y, q.z, q.w]

            # Convert quaternion → Euler angles (in radians)
            roll, pitch, yaw = Rotation_test.from_quat(rotation).as_euler('xyz', degrees=False)

            th = yaw
            # current position
            p = [self.position.x,self.position.y]

            # error in world frame -> car frame
            v = [wp_1_mod[0]-p[0],wp_1_mod[1]-p[1]]
            # speed_controller = 1*speed_command*(v[0] * np.cos(th) + v[1] * np.sin(th))
            R = np.array([[np.cos(th), -np.sin(th)],[np.sin(th),np.cos(th)]])

            v_car = v@R

            # pursuit geometry
            WaypointDist = np.linalg.norm(v_car)
            psi = np.arctan2(v_car[1],v_car[0])


            # pure pursuit algorithm
            delta = np.arctan2(2*self.L*np.sin(psi),WaypointDist) 
            # distance to waypoint (world)
            dist = np.linalg.norm([p[0]-wp_1_mod[0],p[1]-wp_1_mod[1]])

            Kp_steering = 2

            steering = np.clip(
                          Kp_steering*delta,
                          -self.max_steering_angle,
                          self.max_steering_angle)

            self.desired_steering = steering
            if (dist <0.1) and self.K == 0:
              self.FSM = 2
              self.get_logger().info("Reached middle point, proceeding to do the circle")
        if self.FSM == 2:
          w_set = speed_command / self.radius
          self.desired_steering = np.arctan2(self.L*w_set,speed_command)   # pure pursuit for circle
          number_of_steps = 790
          self.K +=1
          if self.K >= number_of_steps:
            self.FSM = 3
            self.K = 0
            self.get_logger().info("Completed circle, proceeding to end point")
        if self.FSM == 3:
          self.desired_steering = 0.0
          number_of_steps = 150
          self.K +=1
          if self.K >= number_of_steps:
            self.FSM = 4
            self.K = 0
            self.desired_speed = 0.0
            self.desired_steering = 0.0

            self.get_logger().info("Reached end point, trajectory finished")
        
      else:
        enable = 0
        speed_command = 0.0
        self.desired_steering = 0.0

        
        
        # publishing commands
      self.nav_command(enable, speed_command)
      self.publish_steering_current()
        

    def nav_command(self,enable, speed_command):
      QCarCommands = Twist()
      QCarCommands.linear.x = enable*speed_command
      QCarCommands.angular.z = enable*self.desired_steering
      self.publisher.publish(QCarCommands)
      
    def publish_steering_current(self):
      steering_msg = Float32()
      a_phi = 1.0 - np.exp(-self.dt / self.steering_constant)
      self.current_steering = self.current_steering + a_phi * (self.desired_steering - self.current_steering)
      steering_msg.data = self.current_steering
      self.publisher_steering.publish(steering_msg)

       
def main():

  # Start the ROS 2 Python Client Library
  rclpy.init()

  node = LeaderTrajectory()
  try:
      rclpy.spin(node)
  except KeyboardInterrupt:
      pass

  rclpy.shutdown()

if __name__ == '__main__':
  main()