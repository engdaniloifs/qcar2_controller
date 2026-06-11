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
from std_msgs.msg import String
from rclpy.qos import QoSProfile, ReliabilityPolicy
from scipy.spatial.transform import Rotation as Rotation_test
from actions_demo_interfaces.msg import FlagsRobotsArray
from std_msgs.msg import Bool, Float32
from pathlib import Path

class pure_pursuit(Node):

    def __init__(self):
      super().__init__('pure_pursuit')

       # --- Parameters ---
      self.declare_parameter('qcarnumber', 1)
      self.qcarnumber = self.get_parameter('qcarnumber').get_parameter_value().integer_value

      self.declare_parameter('config_dir', '')
      self.config_dir = self.get_parameter('config_dir').get_parameter_value().string_value
      if self.config_dir == '':
          raise ValueError('config_dir parameter was not provided')

      self.desired_speed = 0.0


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
      
      # --- Control/algorithm settings ---
      self.wpi = 0

      self.dt = 1/80 # control period

      self.FSM = 0

      self.wp  = np.array([0.0, 0.0, 0.0]) # target waypoint (x,y,z)
      self.desired_steering =0.0 # steering command
      self.current_steering = 0.0 # steering feedback (if available)
      self.steering_constant = 0.16 # steering time constant for first order filter

      self.path_control_timer = self.create_timer(self.dt, self.path_planner) # main loop

      self.path_complete = False # completion flag

      self.publisher = self.create_publisher(Twist,'cmd_vel_nav', 1) # cmd pub
      self.publisher_steering = self.create_publisher(Float32,'current_steering_angle', 10) # steering pub (if needed)
      self.max_steering_angle = 0.6 # steering limit [rad]

      self.mode = 0 # task mode received from FlagsRobotsArray
      self.status = f'done {self.mode}'

      self.config_dir = Path(self.config_dir)
      csv_path = self.config_dir / "path.csv"

        # Read waypoints immediately at node startup
      self.waypoints = self.read_waypoints(csv_path)
      self.get_logger().info(f"Loaded {len(self.waypoints)} waypoints from {csv_path}")

      # --- Subscriptions ---
      # mocap pose (remapped in launch)
      self.subscription_vycon = self.create_subscription(
                                                          PoseStamped,
                                                          'vrpn_pose',  # relative, neutral name
                                                          self.pose_vycon_callback,
                                                          QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, depth=10)
                                                        )
      
      # task manager flags (array of robot flags)
      
      self.subscription_stop_flag = self.create_subscription(Bool, '/qcar/stop',self.stop_experiment_callback , 10)
      # status publisher
      topic1 = f'/Qcar2_{self.qcarnumber}/mission_status'
      self.path_status_publisher = self.create_publisher(String, topic1, 10)

    def stop_experiment_callback(self, msg: Bool):
      self.FSM = msg.data
      if not self.FSM:
        self.get_logger().info("User called STOP ")
        self.flag1 = False
      else:
        self.get_logger().info("User called START")
        self.flag1 = True

    # mocap pose callback
    def pose_vycon_callback(self,msg):
      self.position = msg.pose.position
      self.orientation = msg.pose.orientation

    
    # main control loop
    def read_waypoints(self, filename):
        waypoints = []
        try:
            with open(filename, 'r') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    x = float(row['x'])
                    y = float(row['y'])
                    speed = float(row['speed'])
                    waypoints.append((x, y, speed))
        except Exception as e:
            self.get_logger().error(f"Error reading {filename}: {e}")
        return waypoints

    

    def path_planner(self):

        enable = 1
        speed_command = self.desired_speed

   
        try:
          if self.FSM == 1:

            # --- target extraction ---
            
            robotx, roboty, self.desired_speed = self.waypoints[self.wpi]
            wp_1 = np.array([robotx,roboty])


            wp_1_mod = [0,0]


            wp_1_mod = wp_1

            L= 0.256 # wheelbase

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
            R = np.array([[np.cos(th), -np.sin(th)],[np.sin(th),np.cos(th)]])

            v_car = v@R

            # pursuit geometry
            WaypointDist = np.linalg.norm(v_car)
            psi = np.arctan2(v_car[1],v_car[0])


            # pure pursuit algorithm
            delta = np.arctan2(2*L*np.sin(psi),WaypointDist) 
            # distance to waypoint (world)
            dist = np.linalg.norm([p[0]-wp_1_mod[0],p[1]-wp_1_mod[1]])

            # stop when close
            if ((dist <0.1 ) and (self.wpi != (len(self.waypoints)-1))):
              self.wpi += 1
            if((dist <0.05) and (self.wpi == (len(self.waypoints)-1))):
               speed_command = 0
               steering = 0
               self.path_complete = True 

            # steering P-gain and saturation
            Kp_steering = 2

            steering = np.clip(
                          Kp_steering*delta,
                          -self.max_steering_angle,
                          self.max_steering_angle)

            self.desired_steering = steering
            
        except KeyboardInterrupt:
          speed_command = 0.0
          steering = 0.0
        # enables (based on flags/completion)

        if self.FSM == 1:
          enable = 1.0
        if self.path_complete or self.FSM == 0:
          enable = 0.0

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

  node = pure_pursuit()
  try:
      rclpy.spin(node)
  except KeyboardInterrupt:
      pass

  rclpy.shutdown()

if __name__ == '__main__':
  main()