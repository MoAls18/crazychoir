import numpy as np
from collections import namedtuple
from threading import Event
from choirbot.optimizer.cbba_optimizer import CBBAOptimizer
from choirbot.communicator import StaticCommunicator
import rclpy

Task = namedtuple('Task', ['id', 'coordinates', 'value', 'seq_num'])
class TaskList:
    def __init__(self, tasks):
        self.tasks = tasks

class MockCommunicator(StaticCommunicator):
    def __init__(self):
        self.agents = {}
        self.message_queues = {}
        super().__init__(agent_id=0, size=2, in_neighbors=[])
    def register_agent(self, agent_id, agent):
        self.agents[agent_id] = agent
        self.message_queues[agent_id] = []
    
    def neighbors_exchange(self, message, in_neighbors, out_neighbors, blocking=True, halt_event=None):
            sender_id = message['agent_id']
            responses = {}

            # Step 1: Send message to all out_neighbors
            for neighbor_id in out_neighbors:
                if neighbor_id in self.agents:
                    # print(f"Agent {sender_id} sending message to neighbor {neighbor_id}")
                    self.message_queues[neighbor_id].append((sender_id, message.copy()))
            
            # Step 2: Check for responses from in_neighbors
            for neighbor_id in in_neighbors:
                if neighbor_id in self.agents:
                    # Find any message from this neighbor
                    for i, (msg_sender, msg) in enumerate(self.message_queues[sender_id]):
                        if msg_sender == neighbor_id:
                            # Found a message from this neighbor
                            responses[neighbor_id] = msg
                            # Remove it to avoid processing it again
                            self.message_queues[sender_id].pop(i)
                            break
            
            # print(f"Agent {sender_id} received responses: {responses}")
            return responses

class MockGuidance:
    def __init__(self, agent_id, communicator, position):
        self.agent_id = agent_id
        self.in_neighbors = []
        self.out_neighbors = []
        self.communicator = communicator
        self.current_pose = namedtuple('Pose', ['position'])
        self.current_pose.position = np.array(position + [0.0])
        self.default_velocity = 1.0
    
    def get_logger(self):
        class MockLogger:
            def info(self, message):
                print("INFO:", message)
            def error(self, message):
                print("ERROR:", message)
            def warn(self, message):
                print("WARN:", message)
        return MockLogger()

def test_two_agents():
    rclpy.init()
    print("Starting test_two_agents.")
    random_generator = np.random.default_rng(10)
    
    tasks = []
    
    
    for i in range(1,20):
        tasks.append(Task(id=i, coordinates=np.array([random_generator.uniform(-20, 20), random_generator.uniform(-20, 20)]), value=int(random_generator.integers(5)), seq_num=i))

    task_list = TaskList(tasks)
    communicator = MockCommunicator()
    
    agent1_id, agent2_id, agent3_id, agent4_id, agent5_id = 1, 2, 3, 4, 5
    agent1_position = [3.0, 1.0]
    agent2_position = [-2.0, -1.0]
    agent3_position = [-1.5, 2.0]
    agent4_position = [-2.5, 2.0]
    agent5_position = [4.0, 4.0]

    guidance1 = MockGuidance(agent1_id, communicator, agent1_position)
    guidance2 = MockGuidance(agent2_id, communicator, agent2_position)
    guidance3 = MockGuidance(agent3_id, communicator, agent3_position)
    guidance4 = MockGuidance(agent4_id, communicator, agent4_position)
    guidance5 = MockGuidance(agent5_id, communicator, agent5_position)
    
    guidance1.in_neighbors = [guidance2.agent_id, guidance3.agent_id, guidance4.agent_id, guidance5.agent_id]
    guidance1.out_neighbors = [guidance2.agent_id, guidance3.agent_id, guidance4.agent_id, guidance5.agent_id]
    guidance2.in_neighbors = [guidance1.agent_id, guidance3.agent_id, guidance4.agent_id, guidance5.agent_id]
    guidance2.out_neighbors = [guidance1.agent_id, guidance3.agent_id, guidance4.agent_id, guidance5.agent_id]
    guidance3.in_neighbors = [guidance1.agent_id, guidance2.agent_id, guidance4.agent_id, guidance5.agent_id]
    guidance3.out_neighbors = [guidance1.agent_id, guidance2.agent_id, guidance4.agent_id, guidance5.agent_id]
    guidance4.in_neighbors = [guidance1.agent_id, guidance2.agent_id, guidance3.agent_id, guidance5.agent_id]
    guidance4.out_neighbors = [guidance1.agent_id, guidance2.agent_id, guidance3.agent_id, guidance5.agent_id]
    guidance5.in_neighbors = [guidance1.agent_id, guidance2.agent_id, guidance3.agent_id, guidance4.agent_id]
    guidance5.out_neighbors = [guidance1.agent_id, guidance2.agent_id, guidance3.agent_id, guidance4.agent_id]
    
    max_depth = 2 * len(tasks) / (len(guidance1.in_neighbors) + 1) + 2
    optimizer1 = CBBAOptimizer({'max_bundle_size': max_depth, 'task_value_weight': 0.9})
    optimizer2 = CBBAOptimizer({'max_bundle_size': max_depth, 'task_value_weight': 0.9})
    optimizer3 = CBBAOptimizer({'max_bundle_size': max_depth, 'task_value_weight': 0.9})
    optimizer4 = CBBAOptimizer({'max_bundle_size': max_depth, 'task_value_weight': 0.9})
    optimizer5 = CBBAOptimizer({'max_bundle_size': max_depth, 'task_value_weight': 0.9})
    
    optimizer1.initialize(guidance1)
    optimizer2.initialize(guidance2)
    optimizer3.initialize(guidance3)
    optimizer4.initialize(guidance4)
    optimizer5.initialize(guidance5)
    
    communicator.register_agent(guidance1.agent_id, optimizer1)
    communicator.register_agent(guidance2.agent_id, optimizer2)
    communicator.register_agent(guidance3.agent_id, optimizer3)
    communicator.register_agent(guidance4.agent_id, optimizer4)
    communicator.register_agent(guidance5.agent_id, optimizer5)
    
    optimizer1.create_problem(task_list)
    optimizer2.create_problem(task_list)
    optimizer3.create_problem(task_list)
    optimizer4.create_problem(task_list)
    optimizer5.create_problem(task_list)
    
    print("RUNNING OPTIMIZATION.")
    # optimizer1.build_bundle()
    # optimizer1.create_cbba_message()
    # optimizer2.build_bundle()
    # optimizer2.create_cbba_message()
    # optimizer1.optimize()
    # optimizer2.optimize()

    optimizer1.build_bundle()
    optimizer2.build_bundle()
    optimizer3.build_bundle()
    optimizer4.build_bundle()
    optimizer5.build_bundle()
    print(f"Bundle built by agent {agent1_id}: {optimizer1.bundle}")
    print(f"Bundle built by agent {agent2_id}: {optimizer2.bundle}")
    print(f"Bundle built by agent {agent3_id}: {optimizer3.bundle}")
    print(f"Bundle built by agent {agent4_id}: {optimizer4.bundle}")
    print(f"Bundle built by agent {agent5_id}: {optimizer5.bundle}")
        
    print("\n RUNNING CONSENSUS.")
    for i in range(20):
        print(f"Iteration {i+1}")

        print(optimizer1.winner_bids)
        print(optimizer2.winner_bids)
        print(optimizer3.winner_bids)
        print(optimizer4.winner_bids)
        print(optimizer5.winner_bids)
        
        msg1 = optimizer1.create_cbba_message()
        # print(f"Agent {agent1_id} message: {msg1}")
        msg2 = optimizer2.create_cbba_message()
        # print(f"Agent {agent2_id} message: {msg2}")
        msg3 = optimizer3.create_cbba_message()
        # print(f"Agent {agent3_id} message: {msg3}")
        msg4 = optimizer4.create_cbba_message()
        # print(f"Agent {agent4_id} message: {msg4}")
        msg5 = optimizer5.create_cbba_message()
        # print(f"Agent {agent5_id} message: {msg5}")
        
        # Exchange messages
        # Process messages
        # optimizer1.process_cbba_message(guidance2.agent_id, msg2)
        # optimizer2.process_cbba_message(guidance1.agent_id, msg1)

        # communicator.message_queues[agent1_id].append((agent2_id, msg2.copy()))
        # communicator.message_queues[agent1_id].append((agent3_id, msg3.copy()))
        # communicator.message_queues[agent1_id].append((agent4_id, msg4.copy()))
        # communicator.message_queues[agent1_id].append((agent5_id, msg5.copy()))
        # communicator.message_queues[agent2_id].append((agent1_id, msg1.copy()))
        # communicator.message_queues[agent2_id].append((agent3_id, msg3.copy()))
        # communicator.message_queues[agent2_id].append((agent4_id, msg4.copy()))
        # communicator.message_queues[agent2_id].append((agent5_id, msg5.copy()))
        # communicator.message_queues[agent3_id].append((agent1_id, msg1.copy()))
        # communicator.message_queues[agent3_id].append((agent2_id, msg2.copy()))
        # communicator.message_queues[agent3_id].append((agent4_id, msg4.copy()))
        # communicator.message_queues[agent3_id].append((agent5_id, msg5.copy()))
        # communicator.message_queues[agent4_id].append((agent1_id, msg1.copy()))
        # communicator.message_queues[agent4_id].append((agent2_id, msg2.copy()))
        # communicator.message_queues[agent4_id].append((agent3_id, msg3.copy()))
        # communicator.message_queues[agent4_id].append((agent5_id, msg5.copy()))
        # communicator.message_queues[agent5_id].append((agent1_id, msg1.copy()))
        # communicator.message_queues[agent5_id].append((agent2_id, msg2.copy()))
        # communicator.message_queues[agent5_id].append((agent3_id, msg3.copy()))
        # communicator.message_queues[agent5_id].append((agent4_id, msg4.copy()))

        optimizer1.consensus()
        optimizer2.consensus()
        optimizer3.consensus()
        optimizer4.consensus()
        optimizer5.consensus()
        

        optimizer1.build_bundle()
        optimizer2.build_bundle()
        optimizer3.build_bundle()
        optimizer4.build_bundle()
        optimizer5.build_bundle()
        
        print(f"Agent 1 winners: {optimizer1.winners}")
        print(f"Agent 2 winners: {optimizer2.winners}")
        print(f"Agent 3 winners: {optimizer3.winners}")
        print(f"Agent 4 winners: {optimizer4.winners}")
        print(f"Agent 5 winners: {optimizer5.winners}")
        
        all_match = True
        for task_id in optimizer1.winners:
            winner1 = optimizer1.winners[task_id]
            if not (winner1 == optimizer2.winners[task_id] == optimizer3.winners[task_id] == optimizer4.winners[task_id] == optimizer5.winners[task_id]):
                all_match = False
                break
        
        if all_match:
            
            # optimizer1.consensus()
            # optimizer2.consensus()
            # optimizer3.consensus()
            # optimizer4.consensus()
            # optimizer5.consensus()
            print("Consensus reached after iteration: ", i+1)
            # optimizer1.update_path_from_winners()
            # optimizer2.update_path_from_winners()
            break
    
    #Final results
    print(f"\n Final Assignment:")
    print(f"Agent 1 Tasks: {[t.id for t in optimizer1.get_result()]} Agent 1 path: {optimizer1.path} Agent 1 bundle: {optimizer1.bundle}")
    print(f"Agent 2 Tasks: {[t.id for t in optimizer2.get_result()]} Agent 2 path: {optimizer2.path} Agent 2 bundle: {optimizer2.bundle}")
    print(f"Agent 3 Tasks: {[t.id for t in optimizer3.get_result()]} Agent 3 path: {optimizer3.path} Agent 3 bundle: {optimizer3.bundle}")
    print(f"Agent 4 Tasks: {[t.id for t in optimizer4.get_result()]} Agent 4 path: {optimizer4.path} Agent 4 bundle: {optimizer4.bundle}")
    print(f"Agent 5 Tasks: {[t.id for t in optimizer5.get_result()]} Agent 5 path: {optimizer5.path} Agent 5 bundle: {optimizer5.bundle}")

    # for task in optimizer1.path:
    #     print(f"agent 1 path : {task} with score {optimizer1.winner_bids[task]}")
    for task_id in tasks:
        winner1 = optimizer1.winners[task_id.id]
        winner2 = optimizer2.winners[task_id.id]
        winner3 = optimizer3.winners[task_id.id]
        winner4 = optimizer4.winners[task_id.id]
        winner5 = optimizer5.winners[task_id.id]
        assert winner1 == winner2 == winner3 == winner4 == winner5, f"Task {task_id} assigned to different agents: {winner1} vs {winner2} vs {winner3} vs {winner4} vs {winner5}"
    
    print("Two agent test passed successfully.")    

    def visualize_assignment(tasks, agent_positions, assignments, paths):
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 8))
        
        # Plot tasks
        task_x = [t.coordinates[0] for t in tasks]
        task_y = [t.coordinates[1] for t in tasks]
        plt.scatter(task_x, task_y, c='blue', marker='o', label='Tasks')
        
        # Label tasks with IDs and values
        for t in tasks:
            plt.annotate(f"{t.id}(value={t.value:.1f})", 
                        (t.coordinates[0], t.coordinates[1]),
                        textcoords="offset points", 
                        xytext=(0,10), 
                        ha='center')
        
        # Plot agents
        agent_colors = ['red', 'green', 'black', 'orange', 'purple']
        for i, pos in enumerate(agent_positions):
            agent_id = i + 1
            plt.scatter(pos[0], pos[1], c=agent_colors[i], marker='s', 
                    s=100, label=f'Agent {agent_id}')
        
        # Create a dictionary to quickly look up tasks by ID
        task_dict = {t.id: t for t in tasks}
        
        # Draw sequential paths for each agent
        for agent_idx, path in enumerate(paths):
            if not path:  # Skip if path is empty
                continue
                
            # Start from the agent's position
            points_x = [agent_positions[agent_idx][0]]
            points_y = [agent_positions[agent_idx][1]]
            
            # print(path)
            # Add each task position in order
            for task_tup in path:
                task_id = task_tup.id
                task = task_dict[task_id]
                points_x.append(task.coordinates[0])
                points_y.append(task.coordinates[1])
            
            # Draw the complete path
            plt.plot(points_x, points_y, c=agent_colors[agent_idx], linestyle='-', 
                    marker='x', markersize=5, alpha=0.7, 
                    label=f'Agent {agent_idx+1} path')
        
        plt.title('Task Assignment and Path Visualization')
        plt.legend()
        plt.grid(True)
        plt.savefig('assignment_visualization.png')
        plt.show()

    cbba_assignments = {task.id: optimizer1.winners[task.id] for task in tasks}
    visualize_assignment(tasks, [agent1_position, agent2_position,agent3_position, agent4_position, agent5_position], cbba_assignments, [optimizer1.get_result(), optimizer2.get_result(), optimizer3.get_result(), optimizer4.get_result(), optimizer5.get_result()])
    rclpy.shutdown()
if __name__ == "__main__":
    test_two_agents()


