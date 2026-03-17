import rclpy
from rclpy.executors import MultiThreadedExecutor
from collect_path_pkg.car_state_node import CarStateNode
from collect_path_pkg.wheel_vel_node import WheelVelNode
from collect_path_pkg.obstacles_node import ObstaclesNode
from collect_path_pkg.target_vel_node import TargetVelNode
from collect_path_pkg.path_points_node import PathPointsNode
from collect_path_pkg.collect_path_node import CollectPathNode
from collect_path_pkg.planned_points_node import PlannedPointsNode

def main():
    rclpy.init()

    car_state_node = CarStateNode()
    wheel_vel_node = WheelVelNode()
    obstacles_node = ObstaclesNode()
    target_vel_node = TargetVelNode()
    path_points_node = PathPointsNode()
    planned_points_node = PlannedPointsNode()
    collect_path_node = CollectPathNode(
            car_state_node, 
            path_points_node, 
            wheel_vel_node, 
            target_vel_node, 
            planned_points_node,
            obstacles_node
        )

    executor = MultiThreadedExecutor()
    executor.add_node(car_state_node)
    executor.add_node(wheel_vel_node)
    executor.add_node(target_vel_node)
    executor.add_node(path_points_node)
    executor.add_node(planned_points_node)
    executor.add_node(obstacles_node)
    executor.add_node(collect_path_node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()
