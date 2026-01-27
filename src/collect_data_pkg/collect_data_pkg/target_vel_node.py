from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

class TargetVelNode(Node):
    def __init__(self):
        super().__init__('target_vel_node')
        self.front_subscription = self.create_subscription(
            Float64MultiArray,
            '/car_C_front_wheel',
            self.target_front_vel_callback,
            10)
        self.front_subscription  # prevent unused variable warning

        self.rear_subscription = self.create_subscription(
            Float64MultiArray,
            '/car_C_rear_wheel',
            self.target_rear_vel_callback,
            10)
        self.rear_subscription  # prevent unused variable warning

        self.target_front_vels = []
        self.target_rear_vels = []
        self.target_velocities = []
        self.get_logger().info('TargetVelNode has been started.')

    def target_front_vel_callback(self, msg):
        self.target_front_vels = msg.data  # Assuming data is a list of velocities

        # self.get_logger().info(f'Target Front Velocities: {self.target_front_vels}')

    def target_rear_vel_callback(self, msg):
        self.target_rear_vels = msg.data  # Assuming data is a list of velocities

        # self.get_logger().info(f'Target Rear Velocities: {self.target_rear_vels}')