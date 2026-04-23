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
        data = msg.data # [flag, x, y]
        if len(data) >= 30:
            for i in range(0, 10):
                setattr(self, f'obstacle_{i}', Obstacle(flag=data[i*3], x=data[i*3 + 1], y=data[i*3 + 2]))
            self.obstacles = [getattr(self, f'obstacle_{i}') for i in range(10)]
        # self.get_logger().info(f'Received Obstacles Data: {self.obstacles}')


class Obstacle:
    def __init__(self, flag=0, x=0.0, y=0.0):
        self.flag = flag
        self.x = x
        self.y = y