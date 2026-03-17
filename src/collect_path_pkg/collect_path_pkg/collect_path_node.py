import os
import csv
import numpy as np
from rclpy.node import Node

class CollectPathNode(Node):
    def __init__(self, car_state_node, path_points_node, wheel_vel_node, target_vel_node, planned_points_node, obstacles_node):
        super().__init__('collect_path_node')
        self.car_state_node = car_state_node
        self.path_points_node = path_points_node
        self.wheel_vel_node = wheel_vel_node
        self.target_vel_node = target_vel_node
        self.planned_points_node = planned_points_node
        self.obstacles_node = obstacles_node
        self.get_logger().info('CollectPathNode has been started.')

        home_dir = os.path.expanduser('~')
        self.filepath = os.path.join(home_dir, 'CarController_Isaac', 'pathData.csv')
        with open(self.filepath, mode='w', newline='') as file:
            writer = csv.writer(file)
            header = [
                'vel_left', 'vel_right',
                'pos_x', 'pos_y', 'angle'
            ]
            for i in range(10):
                header.extend([f'target_x_{i}', f'target_y_{i}', f'target_angle_{i}'])
            for i in range(5):
                header.extend([f'obstacle_x_{i}', f'obstacle_y_{i}', f'obstacle_radius_{i}'])
            for i in range(3):
                header.extend([f'planned_x_{i}', f'planned_y_{i}', f'planned_angle_{i}'])
            writer.writerow(header)

        self.timer = self.create_timer(0.05, self.collect_path_callback)  # 20 Hz

    def collect_path_callback(self):
        position = self.car_state_node.position
        orientation = self.car_state_node.orientation
        nearest_points = self.path_points_node.nearest_points
        wheel_velocities = self.wheel_vel_node.wheel_velocities
        target_velocities = self.target_vel_node.target_velocities
        planned_points = self.planned_points_node.planned_points
        obstacles = self.obstacles_node.obstacles
        if not position:
            self.get_logger().warn('Waiting for car state data...')
            return
        if not orientation:
            self.get_logger().warn('Waiting for car state data...')
            return
        if not nearest_points:
            self.get_logger().warn('Waiting for path points data...')
            return
        if not wheel_velocities:
            self.get_logger().warn('Waiting for wheel velocities data...')
            return
        if not target_velocities:
            self.get_logger().warn('Waiting for target velocities data...')
            return
        if not planned_points:
            self.get_logger().warn('Waiting for planned points data...')
            return
        if not obstacles:
            self.get_logger().warn('Waiting for obstacles data...')
            return

        if position and orientation and nearest_points and wheel_velocities and target_velocities and planned_points and obstacles:
            self.get_logger().info('Collecting path data...')
            pos_x = position.x
            pos_y = position.y
            angle = self.compute_angle(
                orientation.x,
                orientation.y,
                orientation.z,
                orientation.w
            )

            for i in range(10):
                setattr(self, f'target_x_{i}', nearest_points[i].x)
                setattr(self, f'target_y_{i}', nearest_points[i].y)
                setattr(self, f'target_angle_{i}', nearest_points[i].angle)

            for i in range(5):
                setattr(self, f'obstacle_x_{i}', obstacles[i].x)
                setattr(self, f'obstacle_y_{i}', obstacles[i].y)
                setattr(self, f'obstacle_radius_{i}', obstacles[i].radius)

            for i in range(3):
                setattr(self, f'planned_x_{i}', planned_points[i].x)
                setattr(self, f'planned_y_{i}', planned_points[i].y)
                setattr(self, f'planned_angle_{i}', planned_points[i].angle)

            vel_left = wheel_velocities.get('Revolute_3', 0.0)
            vel_right = wheel_velocities.get('Revolute_4', 0.0)

            # target_vel_left = target_velocities[0]  # Assuming index 0 is left wheel
            # target_vel_right = target_velocities[1]  # Assuming index 1 is right wheel

            with open(self.filepath, mode='a', newline='') as file:
                writer = csv.writer(file)
                row = [
                    vel_left, vel_right,
                    pos_x, pos_y, angle
                ]
                for i in range(10):
                    row.extend([
                        getattr(self, f'target_x_{i}'),
                        getattr(self, f'target_y_{i}'),
                        getattr(self, f'target_angle_{i}')
                    ])
                for i in range(5):
                    row.extend([
                        getattr(self, f'obstacle_x_{i}'),
                        getattr(self, f'obstacle_y_{i}'),
                        getattr(self, f'obstacle_radius_{i}')
                    ])
                for i in range(3):
                    row.extend([
                        getattr(self, f'planned_x_{i}'),
                        getattr(self, f'planned_y_{i}'),
                        getattr(self, f'planned_angle_{i}')
                    ])
                writer.writerow(row)

    def compute_angle(self, qx, qy, qz, qw):
        # Convert quaternion to Euler angles (roll, pitch, yaw)
        siny_cosp = 2 * (qw * qz + qx * qy)
        cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
        radian = np.arctan2(siny_cosp, cosy_cosp)
        angle = np.degrees(radian)

        return angle