import rclpy
from rclpy.executors import MultiThreadedExecutor
from collect_data_pkg.car_state_node import CarStateNode
from collect_data_pkg.wheel_vel_node import WheelVelNode
from collect_data_pkg.target_vel_node import TargetVelNode
from collect_data_pkg.path_points_node import PathPointsNode
from collect_data_pkg.collect_data_node import CollectDataNode

def main():
    rclpy.init()

    car_state_node = CarStateNode()
    wheel_vel_node = WheelVelNode()
    target_vel_node = TargetVelNode()
    path_points_node = PathPointsNode()
    collect_data_node = CollectDataNode(car_state_node, path_points_node, wheel_vel_node, target_vel_node)

    executor = MultiThreadedExecutor()
    executor.add_node(car_state_node)
    executor.add_node(wheel_vel_node)
    executor.add_node(target_vel_node)
    executor.add_node(path_points_node)
    executor.add_node(collect_data_node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()
