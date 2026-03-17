from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

class TargetVelNode(Node):
    def __init__(self):
        super().__init__('target_vel_node')
        self.subscription = self.create_subscription(
            Float64MultiArray,
            '/car_C_front_wheel',
            self.target_vel_callback,
            10)
        self.subscription  # prevent unused variable warning

        self.target_velocities = None
        self.get_logger().info('TargetVelNode has been started.')

    def target_vel_callback(self, msg):
        self.target_velocities = msg.data  # Assuming data is a list of velocities
        # self.get_logger().info(f'Target Velocities: {self.target_velocities}')