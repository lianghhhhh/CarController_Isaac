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
                'target_vel_left_front', 'target_vel_right_front',
                'target_vel_left_rear', 'target_vel_right_rear',
                'vel_left_front', 'vel_right_front',
                'vel_left_rear', 'vel_right_rear',
                'pos_x', 'pos_y', 'angle'
            ]
            writer.writerow(header)

        self.timer = self.create_timer(0.05, self.collect_data_callback)  # 20 Hz

    def collect_data_callback(self):
        position = self.car_state_node.position
        orientation = self.car_state_node.orientation
        # nearest_point = self.path_points_node.nearest_point
        wheel_velocities = self.wheel_vel_node.wheel_velocities
        target_front_vels = self.target_vel_node.target_front_vels
        target_rear_vels = self.target_vel_node.target_rear_vels
        if not position:
            self.get_logger().warn('Waiting for car state data...')
            return
        if not orientation:
            self.get_logger().warn('Waiting for car state data...')
            return
        # if not nearest_point:
        #     self.get_logger().warn('Waiting for path points data...')
        #     return
        if not wheel_velocities:
            self.get_logger().warn('Waiting for wheel velocities data...')
            return
        if not target_front_vels or not target_rear_vels:
            self.get_logger().warn('Waiting for target velocities data...')
            return

        if position and orientation and wheel_velocities and target_front_vels and target_rear_vels:
            self.get_logger().info('Collecting data...')
            pos_x = position.x
            pos_y = position.y
            angle = self.compute_angle(
                orientation.x,
                orientation.y,
                orientation.z,
                orientation.w
            )

            # target_x = nearest_point.x
            # target_y = nearest_point.y
            # target_angle = nearest_point.angle

            vel_left_front = wheel_velocities.get('Revolute_3', 0.0)
            vel_right_front = wheel_velocities.get('Revolute_4', 0.0)
            vel_left_rear = wheel_velocities.get('Revolute_1', 0.0)
            vel_right_rear = wheel_velocities.get('Revolute_2', 0.0)

            target_vel_left_front = target_front_vels[0]  # Assuming index 0 is left front wheel
            target_vel_right_front = target_front_vels[1]  # Assuming index 1 is right front wheel
            target_vel_left_rear = target_rear_vels[0]  # Assuming index 0 is left rear wheel
            target_vel_right_rear = target_rear_vels[1]  # Assuming index 1 is right rear wheel

            with open(self.filepath, mode='a', newline='') as file:
                writer = csv.writer(file)
                row = [
                    target_vel_left_front, target_vel_right_front,
                    target_vel_left_rear, target_vel_right_rear,
                    vel_left_front, vel_right_front,
                    vel_left_rear, vel_right_rear,
                    pos_x, pos_y, angle
                ]
                writer.writerow(row)

    def compute_angle(self, qx, qy, qz, qw):
        # Convert quaternion to Euler angles (roll, pitch, yaw)
        siny_cosp = 2 * (qw * qz + qx * qy)
        cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
        radian = np.arctan2(siny_cosp, cosy_cosp)
        angle = np.degrees(radian)

        return angle