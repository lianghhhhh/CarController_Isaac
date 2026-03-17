from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

class ObstaclesNode(Node):
    def __init__(self):
        super().__init__('obstacles_node')
        self.subscription = self.create_subscription(
            Float32MultiArray,
            '/obstacles',
            self.obstacles_callback,
            10)
        self.subscription  # prevent unused variable warning

        self.obstacles = None  # Will hold the received obstacles data
        self.get_logger().info('ObstaclesNode has been started.')

    def obstacles_callback(self, msg):
        # Store the received obstacles data
        data = msg.data # [x, y, radius]
        if len(data) >= 15:
            for i in range(0, 5):
                setattr(self, f'obstacle_{i}', Obstacle(x=data[i*3], y=data[i*3 + 1], radius=data[i*3 + 2]))
            self.obstacles = [getattr(self, f'obstacle_{i}') for i in range(5)]
        # self.get_logger().info(f'Received Obstacles Data: {self.obstacles}')


class Obstacle:
    def __init__(self, x=0.0, y=0.0, radius=0.0):
        self.x = x
        self.y = y
        self.radius = radius