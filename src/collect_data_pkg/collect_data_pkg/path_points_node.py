from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

class PathPointsNode(Node):
    def __init__(self):
        super().__init__('path_points_node')
        self.subscription = self.create_subscription(
            Float32MultiArray,
            '/nearest_curve_point',
            self.path_points_callback,
            10)
        self.subscription  # prevent unused variable warning

        self.nearest_point = None
        self.get_logger().info('PathPointsNode has been started.')

    def path_points_callback(self, msg):
        # Store the received path point
        data = msg.data # [idx, x, y, angle]
        if len(data) >= 4:
            self.nearest_point = NearestPoint(x=data[1], y=data[2], angle=data[3])
        # self.get_logger().info(f'Received Nearest Path Point: {self.nearest_point}')

class NearestPoint:
    def __init__(self, x=0.0, y=0.0, angle=0.0):
        self.x = x
        self.y = y
        self.angle = angle