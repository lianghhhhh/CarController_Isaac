import os
import torch
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from control_car_nav_path_pkg.pathGenerator import PathGenerator

class GeneratePathNode(Node):
    def __init__(self):
        super().__init__('generate_path_node')
        # self.subscription = self.create_subscription(
        #     Float32MultiArray,
        #     '/path_generation_data',
        #     self.data_callback,
        #     10)
        # self.subscription  # prevent unused variable warning

        # self.path_data = None  # Will hold the received path data
        self.path_generator = None
        self.device = None
        self.init_model()
        self.get_logger().info('GeneratePathNode has been started.')

    # def data_callback(self, msg):
    #     # Store the received path data
    #     self.path_data = msg.data
    #     self.get_logger().info(f'Received Path Data: {self.path_data}')

    def init_model(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        home_dir = os.path.expanduser('~')
        path_generator = PathGenerator()
        path_generator.load_state_dict(torch.load(os.path.join(home_dir, 'CarController_Isaac', 'path_generator_6.pth')))
        path_generator.to(device)
        path_generator.eval()
        self.path_generator = path_generator
        self.device = device

    def generate_path(self, path_data):
        input_tensor = torch.tensor(path_data, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            generated_delta_path = self.path_generator(input_tensor)
        generated_delta_path = generated_delta_path.cpu().numpy().flatten().tolist()

        # scale the generated delta path data
        generated_delta_path = [delta * 0.1 for delta in generated_delta_path]

        # predict delta path data
        return generated_delta_path