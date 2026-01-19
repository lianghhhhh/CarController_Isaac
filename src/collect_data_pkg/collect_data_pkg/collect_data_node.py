import os
import csv
import numpy as np
from rclpy.node import Node

class CollectDataNode(Node):
    def __init__(self, car_state_node, path_points_node, wheel_vel_node, target_vel_node):
        super().__init__('collect_data_node')
        self.car_state_node = car_state_node
        self.path_points_node = path_points_node
        self.wheel_vel_node = wheel_vel_node
        self.target_vel_node = target_vel_node
        self.get_logger().info('CollectDataNode has been started.')

        home_dir = os.path.expanduser('~')
        self.filepath = os.path.join(home_dir, 'CarController_Isaac', 'carData.csv')
        with open(self.filepath, mode='w', newline='') as file:
            writer = csv.writer(file)
            header = [
                'target_vel_left', 'target_vel_right',
                'vel_left', 'vel_right',
                'pos_x', 'pos_y', 'angle',
                'target_x', 'target_y', 'target_angle'
            ]
            writer.writerow(header)

        self.timer = self.create_timer(0.05, self.collect_data_callback)  # 20 Hz

    def collect_data_callback(self):
        position = self.car_state_node.position
        orientation = self.car_state_node.orientation
        nearest_point = self.path_points_node.nearest_point
        wheel_velocities = self.wheel_vel_node.wheel_velocities
        target_velocities = self.target_vel_node.target_velocities
        if not position:
            self.get_logger().warn('Waiting for car state data...')
            return
        if not orientation:
            self.get_logger().warn('Waiting for car state data...')
            return
        if not nearest_point:
            self.get_logger().warn('Waiting for path points data...')
            return
        if not wheel_velocities:
            self.get_logger().warn('Waiting for wheel velocities data...')
            return
        if not target_velocities:
            self.get_logger().warn('Waiting for target velocities data...')
            return

        if position and orientation and nearest_point and wheel_velocities and target_velocities:
            self.get_logger().info('Collecting data...')
            pos_x = position.x
            pos_y = position.y
            angle = self.compute_angle(
                orientation.x,
                orientation.y,
                orientation.z,
                orientation.w
            )

            target_x = nearest_point.x
            target_y = nearest_point.y
            target_angle = nearest_point.angle

            vel_left = wheel_velocities.get('Revolute_3', 0.0)
            vel_right = wheel_velocities.get('Revolute_4', 0.0)

            target_vel_left = target_velocities[0]  # Assuming index 0 is left wheel
            target_vel_right = target_velocities[1]  # Assuming index 1 is right wheel

            with open(self.filepath, mode='a', newline='') as file:
                writer = csv.writer(file)
                row = [
                    target_vel_left, target_vel_right,
                    vel_left, vel_right,
                    pos_x, pos_y, angle,
                    target_x, target_y, target_angle
                ]
                writer.writerow(row)

    def compute_angle(self, qx, qy, qz, qw):
        # Convert quaternion to Euler angles (roll, pitch, yaw)
        siny_cosp = 2 * (qw * qz + qx * qy)
        cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
        radian = np.arctan2(siny_cosp, cosy_cosp)
        angle = np.degrees(radian)

        return angle