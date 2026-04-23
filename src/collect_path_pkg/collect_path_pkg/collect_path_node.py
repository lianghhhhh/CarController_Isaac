import os
import csv
import math
import numpy as np
from rclpy.node import Node
from collections import deque

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

        self.data_index = 0
        self.delay_buffer = deque(maxlen=60) # 3 sec data

        home_dir = os.path.expanduser('~')
        self.filepath = os.path.join(home_dir, 'CarController_Isaac', 'pathData_20_5.csv')
        with open(self.filepath, mode='w', newline='') as file:
            writer = csv.writer(file)
            header = [
                'data_index',
                'vel_left', 'vel_right',
                'pos_x', 'pos_y', 'angle'
            ]
            for i in range(20):
                header.extend([f'target_x_{i}', f'target_y_{i}', f'target_angle_{i}'])
            for i in range(10):
                header.extend([f'obstacle_flag_{i}', f'obstacle_x_{i}', f'obstacle_y_{i}'])
            for i in range(20):
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
            # is_bad_data = False # detect if is bad data

            pos_x = position.x
            pos_y = position.y

            # # check if collision happens
            # if obstacles and not is_bad_data:
            #     for obstacle in obstacles:
            #         if obstacle.flag == 1.0:  # Only consider valid obstacles
            #             distance = math.hypot(pos_x - obstacle.x, pos_y - obstacle.y)
            #             if distance < 0.4:  # Assuming 0.6m is the collision threshold (car width + obs size)
            #                 is_bad_data = True
            #                 self.get_logger().warn(f'Collision detected with obstacle at ({obstacle.x}, {obstacle.y}), distance: {distance:.2f}m.')
            #                 break
            
            # # handle bad data
            # if is_bad_data:
            #     self.delay_buffer.clear()
            #     self.data_index += 1
            #     return

            angle = self.compute_angle(
                orientation.x,
                orientation.y,
                orientation.z,
                orientation.w
            )

            vel_left = wheel_velocities.get('Revolute_3', 0.0)
            vel_right = wheel_velocities.get('Revolute_4', 0.0)

            # target_vel_left = target_velocities[0]  # Assuming index 0 is left wheel
            # target_vel_right = target_velocities[1]  # Assuming index 1 is right wheel

            row = [
                self.data_index,
                vel_left, vel_right,
                pos_x, pos_y, angle
            ]

            for i in range(20):
                row.extend([
                    nearest_points[i].x,
                    nearest_points[i].y,
                    nearest_points[i].angle
                ])
            
            # if has no valid obstacle, randomly skip this data and do not add to buffer
            obs_count = 0
            for i in range(10):
                if obstacles[i].flag == 1.0:  # Only consider valid obstacles
                    obs_count += 1
                row.extend([
                    obstacles[i].flag,
                    obstacles[i].x,
                    obstacles[i].y
                ])
            if obs_count == 0 and np.random.rand() < 0.5:  # 50% chance to skip data without valid obstacles
                return
            
            for i in range(20):
                row.extend([
                    planned_points[i].x,
                    planned_points[i].y,
                    planned_points[i].angle
                ])

            # add every row to buffer, and write to csv when buffer is full
            self.delay_buffer.append(row)
            if len(self.delay_buffer) == self.delay_buffer.maxlen:
                safe_row = self.delay_buffer.popleft()
                with open(self.filepath, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(safe_row)


    def compute_angle(self, qx, qy, qz, qw):
        # Convert quaternion to Euler angles (roll, pitch, yaw)
        siny_cosp = 2 * (qw * qz + qx * qy)
        cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
        radian = np.arctan2(siny_cosp, cosy_cosp)
        angle = np.degrees(radian)

        return angle