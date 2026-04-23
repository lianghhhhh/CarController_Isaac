import rclpy
from rclpy.executors import MultiThreadedExecutor
from control_car_nav_data_pkg.car_state_node import CarStateNode
from control_car_nav_data_pkg.wheel_vel_node import WheelVelNode
from control_car_nav_data_pkg.path_points_node import PathPointsNode
from control_car_nav_data_pkg.control_car_node import ControlCarNode
from control_car_nav_data_pkg.dual_control_node import DualControlNode
from control_car_nav_data_pkg.obstacles_node import ObstaclesNode
from control_car_nav_data_pkg.planned_points_node import PlannedPointsNode

def main():
    rclpy.init()

    car_state_node = CarStateNode()
    wheel_vel_node = WheelVelNode()
    path_points_node = PathPointsNode()
    planned_points_node = PlannedPointsNode()
    obstacles_node = ObstaclesNode()
    control_car_node = ControlCarNode(car_state_node, path_points_node, wheel_vel_node, planned_points_node, obstacles_node)
    # dual_control_node = DualControlNode(car_state_node, path_points_node, wheel_vel_node)

    executor = MultiThreadedExecutor()
    executor.add_node(car_state_node)
    executor.add_node(wheel_vel_node)
    executor.add_node(path_points_node)
    executor.add_node(planned_points_node)
    executor.add_node(obstacles_node)

    executor.add_node(control_car_node)
    # executor.add_node(dual_control_node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()
