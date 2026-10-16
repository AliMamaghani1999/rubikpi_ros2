from launch import LaunchDescription
from launch.actions import RegisterEventHandler, Shutdown
from launch.event_handlers import OnProcessExit
from launch_ros.actions import Node


def generate_launch_description():
    motor_node = Node(
        package='robot_control',
        executable='motor_control',
        name='motor_controller_node',
        output='screen',
        emulate_tty=True,
        parameters=[],
    )

    waypoint_node = Node(
        package='robot_control',
        executable='waypoint_control',
        name='waypoint_controller_node',
        output='screen',
        emulate_tty=True,
        parameters=[],
    )

    # If the waypoint node exits (path finished or Ctrl-C), bring the whole system down.
    shutdown_when_waypoints_done = RegisterEventHandler(
        OnProcessExit(
            target_action=waypoint_node,
            on_exit=[Shutdown(reason='Waypoint controller exited')]
        )
    )

    # If the motor controller exits for any reason, also bring everything down.
    shutdown_when_motor_exits = RegisterEventHandler(
        OnProcessExit(
            target_action=motor_node,
            on_exit=[Shutdown(reason='Motor controller exited')]
        )
    )

    return LaunchDescription([
        motor_node,
        waypoint_node,
        shutdown_when_waypoints_done,
        shutdown_when_motor_exits,
    ])
