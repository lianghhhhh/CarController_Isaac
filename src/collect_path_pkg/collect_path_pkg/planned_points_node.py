from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

class PlannedPointsNode(Node):
    def __init__(self):
        super().__init__('planned_points_node')
        self.subscription = self.create_subscription(
            Float32MultiArray,
            '/planned_waypoints',
            self.planned_points_callback,
            10)
        self.subscription  # prevent unused variable warning

        self.planned_points = None  # Will hold the received planned points data
        self.get_logger().info('PlannedPointsNode has been started.')

    def planned_points_callback(self, msg):
        # Store the received planned path point (3 points)
        data = msg.data # [idx, x, y, angle]
        if len(data) >= 10:
            for i in range(0, 3):
                setattr(self, f'point_{i}', NearestPoint(x=data[1 + i*3], y=data[2 + i*3], angle=data[3 + i*3]))
            self.planned_points = [getattr(self, f'point_{i}') for i in range(3)]
        # self.get_logger().info(f'Received Planned Point: {self.planned_point}')

class NearestPoint:
    def __init__(self, x=0.0, y=0.0, angle=0.0):
        self.x = x
        self.y = y
        self.angle = angle