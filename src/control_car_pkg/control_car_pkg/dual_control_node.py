import os
import torch
import numpy as np
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from control_car_pkg.utils import preprocess_data, denormalize, calActualState
from control_car_pkg.velPredictor import VelPredictor
from control_car_pkg.carPredictor import CarPredictor

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

        self.num_samples = 50       # How many variations to test
        self.noise_level = 2.0      # How much to wiggle the wheel speeds (m/s)
        self.alpha = 0.8            # Weight for previous command

        self.init_model()
        self.timer = self.create_timer(0.05, self.control_car_callback)  # 20 Hz

    def init_model(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        home_dir = os.path.expanduser('~')

        vel_predictor = VelPredictor()
        vel_predictor.load_state_dict(torch.load(os.path.join(home_dir, 'CarController_Isaac', 'new_vel_model_multi.pth')))
        vel_predictor.to(device)
        vel_predictor.eval()

        car_predictor = CarPredictor()
        car_predictor.load_state_dict(torch.load(os.path.join(home_dir, 'CarController_Isaac', 'new_car_model.pth')))
        car_predictor.to(device)
        car_predictor.eval()

        self.vel_predictor = vel_predictor
        self.car_predictor = car_predictor
        self.device = device

    def compute_loss(self, predicted, target):
        dist_cost = (predicted[0] - target[0]) ** 2 + (predicted[1] - target[1]) ** 2
        angle_cost = (predicted[2] - target[2]) ** 2
        total_cost = dist_cost + (0.5 * angle_cost)
        return total_cost

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
            candidates = np.tile(predicted_vel, (self.num_samples, 1))
            noise = np.random.uniform(-self.noise_level, self.noise_level, candidates.shape)
            candidates += noise
            candidates[0] = np.array(predicted_vel)  # ensure the first candidate is the original prediction
            best_vel = predicted_vel
            min_error = float('inf')
            for vel in candidates:
                input_vel = [vel[0], vel[1], vel_left, vel_right]
                input_tensor = torch.tensor(input_vel, dtype=torch.float32).to(self.device)
                with torch.no_grad():
                    predicted_delta = self.car_predictor(input_tensor)
                predicted_delta = predicted_delta.detach().cpu().numpy().flatten().tolist()
                predicted_state = calActualState(pos_x, pos_y, angle, predicted_delta)
                error = self.compute_loss(predicted_state, [target_x[0], target_y[0], target_angle[0]])
                if error < min_error:
                    min_error = error
                    best_vel = vel

            msg = Float64MultiArray()
            msg.data = [float(v) for v in best_vel]
            self.rear_wheel_pub.publish(msg)
            self.front_wheel_pub.publish(msg)
            self.get_logger().info(f'Published predicted velocities: {best_vel}')

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