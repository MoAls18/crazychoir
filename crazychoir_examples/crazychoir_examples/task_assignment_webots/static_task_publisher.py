import rclpy
from rclpy.node import Node
import numpy as np


from collections import namedtuple
Task = namedtuple('Task', ['id', 'coordinates', 'value'])
class TaskList():
    def __init__(self, tasks):
        self.tasks = tasks
tasks = []
import numpy.random
random_generator = numpy.random.default_rng(0)

for i in range(1,100):
    tasks.append(Task(id=i, coordinates=np.array([random_generator.uniform(-20, 20), random_generator.uniform(-20, 20)]), value=random_generator.uniform(1,1)))

task_list_topic = '/task_list'
class TaskListPublisher(Node):
    def __init__(self):
        super().__init__('task_list_publisher')
        self.publisher = self.create_publisher(TaskList, task_list_topic, 10)
        self.timer = self.create_timer(1.0, self.publish_task_list)
    
    def publish_task_list(self):
        msg = TaskList(tasks)
        self.publisher.publish(msg)
        self.get_logger().info(f'Published task list: {msg.tasks}')



def main(args=None):
    rclpy.init(args=args)
    static_task_publisher = TaskListPublisher()
    rclpy.spin(static_task_publisher)
    rclpy.shutdown()

if __name__ == '__main__':
    main()