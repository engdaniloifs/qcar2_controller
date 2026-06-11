# This is the launch file that starts up the basic QCar2 nodes

import subprocess

from launch import LaunchDescription
from launch.actions import (ExecuteProcess, LogInfo, RegisterEventHandler, OpaqueFunction, TimerAction,IncludeLaunchDescription, GroupAction, DeclareLaunchArgument)
from launch.substitutions import PathJoinSubstitution, LaunchConfiguration
from launch.event_handlers import (OnProcessExit, OnProcessStart)
from launch.launch_description_sources import AnyLaunchDescriptionSource
from launch_ros.actions import Node, PushRosNamespace
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():

    qcarnumber_arg = DeclareLaunchArgument(
        'qcarnumber',
        default_value='1',
        description='Numeric identifier for this QCar (e.g., 1, 2, 3)'
    )
    qcarnumber = LaunchConfiguration('qcarnumber')

    config_dir = "/home/nvidia/ros2/src/qcar2_controller/config"


    FBLinear = Node(
        package='qcar2_controller',
        executable='FBlinear',
        name='FBlinear',
        parameters=[
        {
            'qcarnumber': qcarnumber,
            'config_dir': config_dir,
        }
        ],
        remappings=[(
            'vrpn_pose',
            ['vrpn_mocap/Qcar2_', qcarnumber, '/pose']   # becomes /qcar2/vrpn_mocap/Qcar2_2/pose under the namespace
        )]
    )
    nav2_qcar2_converter = Node(
        package='qcar2_nodes',
        executable='nav2_qcar2_converter',
        name='nav2_qcar2_converter',
        parameters=[{'qcarnumber': qcarnumber}]
    )

    planner = Node(
        package='qcar2_controller',
        executable='planner',
        name='planner',
        parameters=[
        {
            'qcarnumber': qcarnumber,
            'controller_type': 'FBLinear',
            'config_dir': config_dir,
        }
        ]
    )

    qcar2_hardware = Node(
        package='qcar2_nodes',
        executable='qcar2_hardware',
        name='qcar2_hardware',
        parameters=[{'qcarnumber': qcarnumber}]
    )
    vrpn_client_launch = IncludeLaunchDescription(
        AnyLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('vrpn_mocap'),
                'launch',
                'client.launch.yaml'   
            ])
        ),
        launch_arguments={'server': '192.168.2.15'}.items()   
    )

    group = GroupAction([
        PushRosNamespace(['qcar2_', qcarnumber]),    
        vrpn_client_launch,      
        FBLinear,
        planner,
        nav2_qcar2_converter,
        qcar2_hardware
    ])
# then return LaunchDescription([... , group])

    
        

     
    return LaunchDescription([
        qcarnumber_arg,
        group,                       
    ])
