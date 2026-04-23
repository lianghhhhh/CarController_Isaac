import rclpy
from rclpy.executors import MultiThreadedExecutor
from control_car_nav_path_pkg.car_state_node import CarStateNode
from control_car_nav_path_pkg.wheel_vel_node import WheelVelNode
from control_car_nav_path_pkg.path_points_node import PathPointsNode
from control_car_nav_path_pkg.control_car_node import ControlCarNode
from control_car_nav_path_pkg.dual_control_node import DualControlNode
from control_car_nav_path_pkg.plan_path_node import PlanPathNode
from control_car_nav_path_pkg.obstacles_node import ObstaclesNode
from control_car_nav_path_pkg.generate_path_node import GeneratePathNode

def main():
    rclpy.init()

    car_state_node = CarStateNode()
    wheel_vel_node = WheelVelNode()
    path_points_node = PathPointsNode()
    plan_path_node = PlanPathNode()
    obstacles_node = ObstaclesNode()
    generate_path_node = GeneratePathNode()
    control_car_node = ControlCarNode(car_state_node, path_points_node, wheel_vel_node, plan_path_node, obstacles_node, generate_path_node)
    # dual_control_node = DualControlNode(car_state_node, path_points_node, wheel_vel_node)

    executor = MultiThreadedExecutor()
    executor.add_node(car_state_node)
    executor.add_node(wheel_vel_node)
    executor.add_node(path_points_node)
    executor.add_node(plan_path_node)
    executor.add_node(obstacles_node)
    executor.add_node(generate_path_node)
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
