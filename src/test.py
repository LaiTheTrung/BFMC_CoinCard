import heapq
from graph_data import graph_data
from neighbor_data import neighbor_data
import sys 

class Node:
    def __init__(self, node_id, x, y, neighbors):
        self.node_id = node_id
        self.x = x
        self.y = y
        self.neighbors = neighbors
        self.g = float('inf')  # Initial g value set to infinity
        self.h = 0  # Heuristic value (to be implemented based on your requirements)
        self.parent = None

    def __lt__(self, other):
        return (self.g + self.h) < (other.g + other.h)

def heuristic(node, goal):
    # Example: Euclidean distance as heuristic
    return ((node.x - goal.x)**2 + (node.y - goal.y)**2)/2

def astar(graph, start_id, goal_id):
    start_node = graph[start_id]
    goal_node = graph[goal_id]

    open_set = [start_node]
    closed_set = set()
    while open_set:
        current_node = heapq.heappop(open_set)
        # print(current_node.node_id)
        if current_node.node_id == goal_id:
            # Destination reached, reconstruct the path
            path = []
            while current_node:
                path.append(current_node.node_id)
                current_node = current_node.parent
            return path[::-1]

        closed_set.add(current_node.node_id)
        # print(current_node.node_id)
        # print(current_node.neighbors)
        for neighbor_id in graph[current_node.node_id].neighbors:
            # print(neighbor_id)
            neighbor_node = graph[neighbor_id]
            # print(neighbor_node.node_id)
            if neighbor_node.node_id in closed_set:
                continue

            tentative_g = current_node.g + 1  # Assuming unit cost for each step

            if tentative_g < neighbor_node.g or neighbor_node not in open_set:
                neighbor_node.g = tentative_g
                neighbor_node.h = heuristic(neighbor_node, goal_node)
                neighbor_node.parent = current_node

                if neighbor_node not in open_set:
                    heapq.heappush(open_set, neighbor_node)

if __name__ == "__main__":
    # if len(sys.argv) != 3:
    #     print("Command: python script.py start_node_id goal_node_id")
    #     sys.exit(1)

    start_node_id = '472'
    goal_node_id = '125'

    graph = {node_id: Node(node_id, data['x'], data['y'], neighbor_data['neighbors']) for (node_id, data),(node_id,neighbor_data) in zip(graph_data.items(),neighbor_data.items())}

    path = astar(graph, start_node_id, goal_node_id)

    if path:
        print(f"Path from {start_node_id} to {goal_node_id}: {path}")
    else:
        print(f"No path found from {start_node_id} to {goal_node_id}")
