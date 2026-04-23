import os
import torch
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from control_car_nav_path_pkg.pathPlanner import PathPlanner
from control_car_nav_path_pkg.utils import preprocess_data, denormalize, preprocess_planning_data, calActualState

class PlanPathNode(Node):
    def __init__(self):
        super().__init__('plan_path_node')
        self.path_planner = None
        self.device = None
        self.init_model()

        self.plan_path_pub = self.create_publisher(Float32MultiArray, "predicted_waypoints", 10)

        self.get_logger().info('PlanPathNode has been started.')

    def init_model(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        home_dir = os.path.expanduser('~')
        path_planner = PathPlanner()
        path_planner.load_state_dict(torch.load(os.path.join(home_dir, 'CarController_Isaac', 'path_planner_new_safe.pth')))
        path_planner.to(device)
        path_planner.eval()
        self.path_planner = path_planner
        self.device = device

    def plan_path(self, data):
        input_tensor = torch.tensor(data, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            planned_path = self.path_planner(input_tensor)
        planned_path = planned_path.cpu().numpy().flatten().tolist()
        return planned_path
    
    def publish_planned_path(self, planned_path):
        msg = Float32MultiArray()
        msg.data = planned_path
        self.plan_path_pub.publish(msg)
        self.get_logger().info(f'Published planned path: {planned_path}')