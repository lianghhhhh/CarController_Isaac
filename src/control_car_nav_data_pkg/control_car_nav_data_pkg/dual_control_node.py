import os
import torch
import numpy as np
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from control_car_nav_data_pkg.utils import preprocess_data, denormalize, calActualState
from control_car_nav_data_pkg.velPredictor import VelPredictor
from control_car_nav_data_pkg.carPredictor import CarPredictor

class DualControlNode(Node):
    def __init__(self, car_state_node, path_points_node, wheel_vel_node):
        super().__init__('dual_control_node')
        self.car_state_node = car_state_node
        self.path_points_node = path_points_node
        self.wheel_vel_node = wheel_vel_node
        self.get_logger().info('DualControlNode has been started.')

        self.rear_wheel_pub = self.create_publisher(Float64MultiArray, "car_C_rear_wheel", 10)
        self.front_wheel_pub = self.create_publisher(Float64MultiArray, "car_C_front_wheel", 10)

        self.vel_predictor = None
        self.car_predictor = None
        self.device = None

        self.num_samples = 2000 # How many variations to test
        self.noise_level = 2.0 # How much to wiggle the wheel speeds (m/s)
        self.max_vel = 10.0 # Maximum wheel speed (m/s)
        self.smooth_weight = 0.8 # Weight for previous command
        self.x_weight = 1 # Weight for x error in loss function
        self.y_weight = 10 # Weight for y error in loss function
        self.angle_weight = 0.2 # Weight for angle error in loss function

        self.init_model()
        self.timer = self.create_timer(0.05, self.control_car_callback)  # 20 Hz

    def init_model(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        home_dir = os.path.expanduser('~')

        vel_predictor = VelPredictor()
        vel_predictor.load_state_dict(torch.load(os.path.join(home_dir, 'CarController_Isaac', 'new_vel_model_multi_1_2_2.pth')))
        vel_predictor.to(device)
        vel_predictor.eval()

        car_predictor = CarPredictor()
        car_predictor.load_state_dict(torch.load(os.path.join(home_dir, 'CarController_Isaac', 'new_car_model.pth')))
        car_predictor.to(device)
        car_predictor.eval()

        self.vel_predictor = vel_predictor
        self.car_predictor = car_predictor
        self.device = device

    def compute_loss(self, predicted_state, target_state, cand_vel, curr_vel):
        pred_x, pred_y, pred_angle = predicted_state
        targ_x, targ_y, targ_angle = target_state
        
        # 1. Distance errors
        dx = targ_x - pred_x
        dy = targ_y - pred_y
        distance_sq = (dx ** 2) + (dy ** 2)
        distance = np.sqrt(distance_sq)
        
        # Calculate Local Errors for X and Y tracking
        local_error_x = dx * np.cos(-pred_angle) - dy * np.sin(-pred_angle)
        local_error_y = dx * np.sin(-pred_angle) + dy * np.cos(-pred_angle)
        
        x_cost = self.x_weight * (local_error_x ** 2)
        y_cost = self.y_weight * (local_error_y ** 2)
        
        # 2. Heading-to-Target Cost (Pure Pursuit)
        # Only apply this if the target is far enough away to avoid the arctan2 singularity
        if distance > 0.1: 
            diff_angle = np.arctan2(dy, dx) - pred_angle
            diff_angle = (diff_angle + np.pi) % (2 * np.pi) - np.pi
            heading_to_target_cost = self.angle_weight * (diff_angle ** 2)
        else:
            heading_to_target_cost = 0.0
            
        # 3. Smoothness Cost
        smooth_cost = self.smooth_weight * ((cand_vel[0] - curr_vel[0])**2 + (cand_vel[1] - curr_vel[1])**2)
        
        # 4. Speed Reward (To keep the car moving fast)
        speed_reward = 0.5 * (cand_vel[0] + cand_vel[1]) 
        
        return x_cost + y_cost + heading_to_target_cost + smooth_cost + speed_reward

    def control_car_callback(self):
        position = self.car_state_node.position
        orientation = self.car_state_node.orientation
        nearest_points = self.path_points_node.nearest_points
        wheel_velocities = self.wheel_vel_node.wheel_velocities
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

        if position and orientation and nearest_points and wheel_velocities:
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

            data = [
                    vel_left, vel_right,
                    pos_x, pos_y, angle,
                    target_x[0], target_y[0], target_angle[0],
                    target_x[1], target_y[1], target_angle[1],
                    target_x[2], target_y[2], target_angle[2]
                ]
            self.get_logger().info(f'Current data: {data}')
            data = preprocess_data(data)

            input_tensor = torch.tensor(data, dtype=torch.float32).to(self.device)

            with torch.no_grad():
                predicted_vel = self.vel_predictor(input_tensor)
            predicted_vel = predicted_vel.detach().cpu().numpy().flatten().tolist()
            predicted_vel = [float(v)*(-1.0) for v in predicted_vel] # invert velocity direction

            # Generate candidates around the predicted velocity
            base_candidates = np.tile(predicted_vel, (self.num_samples, 1))
            noise = np.random.uniform(-self.noise_level, self.noise_level, base_candidates.shape)
            base_candidates = base_candidates + noise
            base_candidates[0] = np.array(predicted_vel)  # ensure the first candidate is the original prediction

            num_wild = self.num_samples - self.num_samples  # num_base is now self.num_samples, so num_wild is 0

            wild_candidates = np.random.uniform(2.0, self.max_vel, (num_wild, 2))

            candidates = np.vstack([base_candidates, wild_candidates])

            current_vel_block = np.tile([vel_left, vel_right], (self.num_samples, 1))
            batch_input = np.hstack([candidates, current_vel_block])
            batch_tensor = torch.tensor(batch_input, dtype=torch.float32).to(self.device)

            with torch.no_grad():
                predicted_deltas = self.car_predictor(batch_tensor)
            predicted_deltas = predicted_deltas.detach().cpu().numpy()

            best_vel = predicted_vel
            id = 0
            min_error = float('inf')
            # for vel in candidates:
            for i in range(self.num_samples):
                cand_vel = candidates[i]
                delta = predicted_deltas[i]
                pred_state = calActualState(pos_x, pos_y, angle, delta)
                target_state = (target_x[0], target_y[0], target_angle[0])
                error = self.compute_loss(pred_state, target_state, cand_vel, [vel_left, vel_right])
                if error < min_error:
                    min_error = error
                    best_vel = cand_vel
                    id = i

            msg = Float64MultiArray()
            msg.data = [float(v) for v in best_vel]
            self.rear_wheel_pub.publish(msg)
            self.front_wheel_pub.publish(msg)
            self.get_logger().info(f'Published predicted velocities: {best_vel} with id {id}, predicted_vel: {predicted_vel}')

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