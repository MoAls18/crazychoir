import rclpy
from choirbot.guidance.task import TaskGuidance, PositionTaskExecutor
from choirbot.optimizer import TaskOptimizer
from choirbot.optimizer.cbba_optimizer import CBBAOptimizer
from crazychoir.guidance.cbba_guidance import CBBAGuidance
def main():
    rclpy.init()

    # initialize task guidance
    opt_settings = {'max_iterations': 50, 'convergence_threshold': 5, 'task_value_weight': 0.95}
    executor = PositionTaskExecutor()
    optimizer = CBBAOptimizer(settings=opt_settings)
    
    guidance = TaskGuidance(optimizer, executor, None, 'pubsub', 'odom')

    rclpy.spin(guidance)
    rclpy.shutdown()
