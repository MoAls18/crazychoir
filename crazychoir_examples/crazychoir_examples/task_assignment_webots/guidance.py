import rclpy
from choirbot.guidance.task import PositionTaskExecutor, CBBAGuidance
from choirbot.optimizer.cbba_optimizer import CBBAOptimizer

def main():
    rclpy.init()

    # initialize task guidance
    opt_settings = {'max_iterations': 100,
                    'convergence_threshold': 3, 'task_value_weight': 0.95}
    executor = PositionTaskExecutor()
    optimizer = CBBAOptimizer(settings=opt_settings)

    guidance = CBBAGuidance(optimizer, executor, None, 'pubsub', 'odom')

    rclpy.spin(guidance)
    rclpy.shutdown()
