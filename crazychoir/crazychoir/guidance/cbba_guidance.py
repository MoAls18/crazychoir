from collections import namedtuple
from typing import Callable
from rclpy.node import Node
from std_msgs.msg import Empty

from choirbot.guidance import OptimizationGuidance 
from choirbot.guidance.optimization_thread import OptimizationThread
from choirbot.optimizer.cbba_optimizer import CBBAOptimizer

Task = namedtuple('Task', ['id', 'coordinates', 'value', 'seq_num'])
class TaskList():
    def __init__(self, tasks):
        self.tasks = tasks

class CBBAGuidance(OptimizationGuidance):
    """
    Guidance node implementing CBBA for task assignment.
    
    """

    def __init__(self, optimizer: CBBAOptimizer, pose_handler: str=None, pose_topic: str=None, pose_callback: Callable = None):
        super().__init__(optimizer, CBBAOptimizationThread, pose_handler, pose_topic, pose_callback)

        self.task_list = []
        self.task_executor = None
        self.current_task = None
        self.completed_tasks = []
        
        self.opt_trigger_subscription = self.create_subscription(
            Empty, '/optimization_trigger', self.start_optimization, 10)

class CBBAOptimizationThread(OptimizationThread):
    pass