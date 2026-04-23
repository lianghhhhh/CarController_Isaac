import os
import torch
import numpy as np
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from control_car_nav_path_pkg.utils import preprocess_data, denormalize, preprocess_planning_data, calActualState, preprocess_generate_data
from control_car_nav_path_pkg.velPredictor import VelPredictor

class ControlCarNode(Node):
    def __init__(self, car_state_node, path_points_node, wheel_vel_node, plan_path_node, obstacles_node, generate_path_node):
        super().__init__('control_car_node')
        self.car_state_node = car_state_node
        self.path_points_node = path_points_node
        self.wheel_vel_node = wheel_vel_node
        self.plan_path_node = plan_path_node
        self.obstacles_node = obstacles_node
        self.generate_path_node = generate_path_node

        self.get_logger().info('ControlCarNode has been started.')

        self.rear_wheel_pub = self.create_publisher(Float64MultiArray, "car_C_rear_wheel", 10)
        self.front_wheel_pub = self.create_publisher(Float64MultiArray, "car_C_front_wheel", 10)

        self.vel_predictor = None
        self.device = None
        self.turn_model = None
        # self.y_error = 0.0

        self.init_model()
        self.timer = self.create_timer(0.05, self.control_car_callback)  # 20 Hz

    def init_model(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        home_dir = os.path.expanduser('~')

        vel_predictor = VelPredictor()
        vel_predictor.load_state_dict(torch.load(os.path.join(home_dir, 'CarController_Isaac', 'new_vel_model_multi_1_2_2.pth')))
        vel_predictor.to(device)
        vel_predictor.eval()

        turn_model = VelPredictor()
        turn_model.load_state_dict(torch.load(os.path.join(home_dir, 'CarController_Isaac', 'new_vel_model_multi_1_4_2.pth')))
        turn_model.to(device)
        turn_model.eval()

        self.vel_predictor = vel_predictor
        self.device = device
        self.turn_model = turn_model

    def control_car_callback(self):
        position = self.car_state_node.position
        orientation = self.car_state_node.orientation
        nearest_points = self.path_points_node.nearest_points
        wheel_velocities = self.wheel_vel_node.wheel_velocities
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
        if not obstacles:
            self.get_logger().warn('Waiting for obstacles data...')
            return

        if position and orientation and nearest_points and wheel_velocities and obstacles:
            pos_x = position.x
            pos_y = position.y
            angle = self.compute_angle(
                orientation.x,
                orientation.y,
                orientation.z,
                orientation.w
            )

            target_x = []
            target_y = []
            target_angle = []
            for point in nearest_points:
                target_x.append(point.x)
                target_y.append(point.y)
                target_angle.append(np.deg2rad(point.angle))

            vel_left = wheel_velocities.get('Revolute_3', 0.0)
            vel_right = wheel_velocities.get('Revolute_4', 0.0)

            planning_data = [
                    vel_left, vel_right,
                    pos_x, pos_y, angle,
                ]
            for i in range(len(target_x)):
                planning_data.append(target_x[i])
                planning_data.append(target_y[i])
                planning_data.append(target_angle[i])
            for obs in obstacles:
                planning_data.append(obs.flag)
                planning_data.append(obs.x)
                planning_data.append(obs.y)
            self.get_logger().info(f'Current data: {planning_data}')
            # planning_data = preprocess_planning_data(planning_data)
            planning_data = preprocess_generate_data(planning_data)

            # planned_path = self.plan_path_node.plan_path(planning_data)
            generated_delta_path = self.generate_path_node.generate_path(planning_data)
            actual_path = []
            for i in range(0, len(generated_delta_path), 3):
                idx = i // 3
                # temp_x = target_x[idx] + generated_delta_path[i]
                # temp_y = target_y[idx] + generated_delta_path[i+1]
                # temp_angle = target_angle[idx] + generated_delta_path[i+2]
                # temp_angle = (temp_angle + np.pi) % (2 * np.pi) - np.pi  # normalize to [-pi, pi]
                # actual_path.extend([temp_x, temp_y, temp_angle])
                
                # local to global coordinate transformation
                global_dx = generated_delta_path[i] * np.cos(angle) - generated_delta_path[i+1] * np.sin(angle)
                global_dy = generated_delta_path[i] * np.sin(angle) + generated_delta_path[i+1] * np.cos(angle)
                temp_x = target_x[idx] + global_dx
                temp_y = target_y[idx] + global_dy
                temp_angle = target_angle[idx] + generated_delta_path[i+2]
                temp_angle = (temp_angle + np.pi) % (2 * np.pi) - np.pi  # normalize to [-pi, pi]
                actual_path.extend([temp_x, temp_y, temp_angle])

            data = [vel_left, vel_right, pos_x, pos_y, angle]
            data.extend(actual_path[:9]) # include first 3 target points (x, y, angle)
            print(f'Actual path data for prediction: {data}')
            data = preprocess_data(data)
            is_turn = abs(data[0][4]) > np.deg2rad(10)  # Check if the angle difference is greater than 10 degrees

            # vel_data = [vel_left, vel_right]
            # for val in planned_path:
            #     vel_data.append(float(val))
            # vel_data = np.array(vel_data).reshape(1, -1)
            # is_turn = abs(vel_data[0, -1]) > np.deg2rad(10)  # Check if the angle difference is greater than 10 degrees


            input_tensor = torch.tensor(data, dtype=torch.float32).to(self.device)

            with torch.no_grad():
                if is_turn:
                    self.get_logger().info('Turning detected, using turn model for prediction.')
                    predicted_vel = self.turn_model(input_tensor)
                # elif abs(data[0][3]) > 0.5:  # Check local_dy for potential lateral error
                #     self.get_logger().info('Lateral error detected, using turn model for prediction.')
                #     predicted_vel = self.turn_model(input_tensor)
                else:
                    self.get_logger().info('Straight driving detected, using velocity model for prediction.')
                    predicted_vel = self.vel_predictor(input_tensor)
            predicted_vel = predicted_vel.detach().cpu().numpy().flatten().tolist()

            # predicted_vel = denormalize(predicted_vel)
            predicted_vel = [float(v)*(-1.0) for v in predicted_vel] # invert velocity direction

            # publish predicted velocities
            msg = Float64MultiArray()
            msg.data = predicted_vel
            self.rear_wheel_pub.publish(msg)
            self.front_wheel_pub.publish(msg)
            self.get_logger().info(f'Published predicted velocities: {predicted_vel}')

            # publish planned path for visualization
            # planned_data = []
            # for i in range(0, len(generated_delta_path), 3):
            #     temp_data = calActualState(pos_x, pos_y, angle, generated_delta_path[i:i+3])
            #     for val in temp_data:
            #         planned_data.append(val)
            # self.plan_path_node.publish_planned_path(planned_data)
            self.plan_path_node.publish_planned_path(actual_path[:30]) # publish first 10 target points for visualization


            # y_error = data[0][3]  # local_dy
            # if abs(y_error) > self.y_error:
            #     self.y_error = y_error
            # self.get_logger().warning(f'Current y_error: {y_error}, Max y_error: {self.y_error}')

    def compute_angle(self, qx, qy, qz, qw):
        # Convert quaternion to yaw angle in radians
        siny_cosp = 2 * (qw * qz + qx * qy)
        cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
        radian = np.arctan2(siny_cosp, cosy_cosp)
        
        # Flip orientation by 180 degrees
        radian = radian + np.pi
        if radian > np.pi:
            radian -= 2 * np.pi
        
        return radian