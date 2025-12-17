import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("TkAgg")
import osmnx as ox
import pyproj
import numpy as np
import csv
import json
import heapq

from time import time_ns
from mcap.writer import Writer
from skimage.draw import line, disk

GRID_SIZE = 1000
NUM_AGENTS = 100
NUM_STEPS = 10000
USE_FILE_DATA = True
MIN_GREEN = 20
MAX_GREEN = 60
NEIGHBOR_RADIUS = 5


class McapLogger:
    def __init__(self, filename="robots.mcap"):
        self.filename = filename
        self.writer = None
        self.channel_id = None

    def __enter__(self):
        self.stream = open(self.filename, "wb")
        self.writer = Writer(self.stream)
        # start writer
        self.writer.start(profile="x-custom", library="robot-grid-sim")
        # schemaless JSON channel for robot states
        self.channel_id = self.writer.register_channel(
            schema_id=0,                # no schema, schemaless JSON
            topic="robots",
            message_encoding="json",
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.writer is not None:
            self.writer.finish()
        if hasattr(self, "stream"):
            self.stream.close()

    def log_step(self, step, robots):
        msg = {
            "step": step,
            "robots": [
                {
                    "id": r.robot_id,
                    "x": int(r.pos[0]),
                    "y": int(r.pos[1]),
                    "goal_x": int(r.goal[0]),
                    "goal_y": int(r.goal[1]),
                    "finished": bool(r.finished),
                }
                for r in robots
            ],
        }

        now = time_ns()
        self.writer.add_message(
            channel_id=self.channel_id,
            log_time=now,
            publish_time=now,
            data=json.dumps(msg).encode("utf-8"),
        )

class Robot:
    def __init__(self, start, goal, grid, grid_graph, intersection_control, robot_id, stick_prob=0.1):
        self.pos = start
        self.goal = goal
        self.grid_graph = grid_graph
        self.grid = grid
        self.intersection_control = intersection_control
        self.stick_prob = stick_prob
        self.finished = False
        self.path = None
        self.robot_id = robot_id
        
    def compute_path(self, current_time, reservation_table, max_horizon=100):
        start = self.pos
        goal = self.goal

        # neighbors in the road graph (4-dir moves + optional wait)
        def neighbors(pos):
            x, y = pos
            for dx, dy in [(-1,0), (1,0), (0,-1), (0,1), (0,0)]:  # include wait
                nx_, ny_ = x + dx, y + dy
                if (nx_, ny_) in self.grid_graph:
                    yield (nx_, ny_)

        def is_reserved(x, y, t):
            return (x, y, t) in reservation_table

        open_heap = []
        start_state = (start[0], start[1], current_time)
        g = {start_state: 0}
        f0 = self.heuristic(start, goal)
        heapq.heappush(open_heap, (f0, start_state))
        parent = {}

        while open_heap:
            _, (x, y, t) = heapq.heappop(open_heap)

            # goal reached
            if (x, y) == goal:
                path = []
                s = (x, y, t)
                while s in parent:
                    path.append((s[0], s[1]))
                    s = parent[s]
                path.append(start)
                path.reverse()
                self.path = path
                return

            if t - current_time > max_horizon:
                break

            for nx_, ny_ in neighbors((x, y)):
                nt = t + 1

                # vertex collision avoidance
                if is_reserved(nx_, ny_, nt):
                    continue

                # edge swap avoidance: other agent coming into (x, y) at nt
                if (x, y, nt) in reservation_table:
                    other_id = reservation_table[(x, y, nt)]
                    if other_id != self.robot_id and is_reserved(nx_, ny_, t):
                        continue

                ns = (nx_, ny_, nt)
                ng = g[(x, y, t)] + 1
                if ns not in g or ng < g[ns]:
                    g[ns] = ng
                    parent[ns] = (x, y, t)
                    f = ng + self.heuristic((nx_, ny_), goal)
                    heapq.heappush(open_heap, (f, ns))

        # no path found within horizon
        self.path = []
        
    def heuristic(self, pos, goal):
        # Manhattan distance
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])
        
    def is_light_green_at_time(self, light, t):
        # Approximate future state: modulo light.green_time*2
        cycle = light.green_time * 2
        phase = (light.timer + t) % cycle
        return (phase < light.green_time) if light.green else (phase >= light.green_time)
        
    def game_theory_decision(self, grid, other_agents):
        action_space = self.sample_continuous_actions(100)
        others = self.get_visible_surroundings(all_agents)
        
        predictions = self.predict(others)
        
        self_action = random.choice(action_space)
        other_action = [random.choice(action_space) for _ in others]
        
        for _ in range(max_iter):
            best_self = max(action_space, key=lambda a: self.compute_utility(self, a, others))
            all_interacting = others + [self]

            best_other = [max(action_space, key=lambda a: self.compute_utility(current, a, [other for other in all_interacting if other != current])) for current in others]
            
            if best_self == self_action and best_other == other_action:
                break
                
            self_action = best_self
            other_action = best_other
            
        best_move = self_action
        print(best_move)
        return 0
        
    def step(self, t, reservation_table):
        if self.finished:
            return

        # Stick
        if np.random.rand() < self.stick_prob:
            return

        if self.pos == self.goal:
            self.finished = True
            return
            
        nearby_robots = []
        for robot in robots:
            if robot.robot_id == self.robot_id or robot.finished:
                continue
                
            rx, ry = robot.pos
            if (rx - self.pos[0])**2 + (ry - self.pos[1])**2 <= NEIGHBOR_RADIUS**2:
                nearby_robots.append(robot)
                
        if nearby_robots:
            action = game_theory_decision(self, self.grid, nearby_robots)
            
            if action is not None:
                # Interpret action as next cell (nx, ny)
                # If your function returns a direction (dx, dy), adapt this line.
                nx_, ny_ = action

                # Bounds and road check
                h, w = self.grid.shape
                if not (0 <= nx_ < w and 0 <= ny_ < h):
                    return
                if self.grid[ny_, nx_] not in (1, 2):
                    return

                # Traffic light check
                light = self.intersection_control[ny_, nx_]
                if light is not None and not light.green:
                    return

                # Move according to game-theoretic action
                self.pos = (nx_, ny_)
                # Discard path so it will be recomputed next time
                self.path = None
                return

        # Compute path if needed
        if self.path is None or self.pos not in self.path or len(self.path) < 2:
            self.compute_path(t, reservation_table)
        
        if self.path is None or len(self.path) < 2:
            occupied = set()
            x, y = self.pos
            h, w = self.grid.shape
            candidates = []
            
            for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
                nx_, ny_ = x + dx, y + dy
                if 0 <= nx_ < w and 0 <= ny_ < h:
                    if self.grid[ny_, nx_] in (1, 2):  # road or intersection
                        if (nx_, ny_) not in occupied:
                            # prefer cells closer to goal
                            score = self.heuristic((nx_, ny_), self.goal)
                            candidates.append(((nx_, ny_), score))

            if not candidates:
                return  # completely stuck; try again next tick

            # choose neighbor that minimizes heuristic distance to goal
            candidates.sort(key=lambda c: c[1])
            fallback_next = candidates[0][0]
            nx_, ny_ = fallback_next

            # respect traffic lights in fallback too
            light = self.intersection_control[ny_, nx_]
            if light is not None and not light.green:
                return  # still must wait at red

            self.pos = fallback_next
            # discard old path so it will be recomputed next step
            self.path = None
        else:
            next_cell = self.path[1]
            nx_, ny_ = next_cell

            # 🚦 Traffic light enforcement
            light = self.intersection_control[ny_, nx_]
            if light is not None and not light.green:
                return  # wait at red

            # collision avoidance at time t+1
            if (nx_, ny_, t + 1) in reservation_table:
                return  # cell is already reserved; try replan next step

            # reserve next cell and move
            reservation_table[(nx_, ny_, t + 1)] = self.robot_id

            # Move one step
            self.pos = next_cell
            self.path = self.path[1:]  # advance path

class TrafficLight:
    def __init__(self, node, gx, gy, radius):
        self.node = node
        self.gx = gx
        self.gy = gy 
        self.radius = radius
        self.green = np.random.choice([True, False])
        self.timer = np.random.randint(0, MAX_GREEN)
        self.green_time = np.random.randint(MIN_GREEN, MAX_GREEN)
        
    def update(self):
        self.timer += 1
        if self.timer >= self.green_time:
            self.green = not self.green
            self.timer = 0


def build_grid_graph(grid):
    G = nx.Graph()
    h, w = grid.shape

    for y in range(h):
        for x in range(w):
            if grid[y, x] not in (1, 2):
                continue

            for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
                nx_, ny_ = x + dx, y + dy
                if 0 <= nx_ < w and 0 <= ny_ < h:
                    if grid[ny_, nx_] in (1, 2):
                        G.add_edge((x, y), (nx_, ny_), weight=1)

    return G

def build_intersection_control_grid(grid, traffic_lights):
    control = np.empty(grid.shape, dtype=object)
    control[:] = None

    for light in traffic_lights.values():
        rr, cc = disk((light.gy, light.gx), light.radius, shape=grid.shape)
        for y, x in zip(rr, cc):
            control[y, x] = light

    return control

def build_light_lookup(traffic_lights):
    lookup = {}
    for light in traffic_lights.values():
        lookup[(light.gx, light.gy)] = light
    return lookup
    
def create_robots(grid, num_agents):
    road_cells = np.argwhere(grid == 1)
    robots = []
    
    for _ in range(num_agents):
        start_idx = np.random.randint(len(road_cells))
        goal_idx = np.random.randint(len(road_cells))
        
        sy, sx = road_cells[start_idx]
        gy, gx = road_cells[goal_idx]
        
        robots.append(((sx, sy), (gx, gy)))
        
    return robots

def fetch_map_data(location_name):
    ### Fetch OpenStreetMap data for the specified location
    G = ox.graph.graph_from_place(location_name, network_type="drive")
    return G
    
def create_bounding_box(G, x_min, x_max, y_min, y_max):
    nodes_to_remove = []

    ### Sort through the graph and only keep the points within the bounding box defined above
    for node, attrs in G.nodes(data=True):
        x = attrs.get('x', 0.0)
        y = attrs.get('y', 0.0)
        if not (x_min <= x <= x_max and y_min <= y <= y_max):
            nodes_to_remove.append(node)

    G.remove_nodes_from(nodes_to_remove)
    
    return G

def discretize_graph(G, x_min, x_max, y_min, y_max):
    ### Convert this graph to a discretized 2D grid
    grid = np.zeros((GRID_SIZE+1, GRID_SIZE+1), dtype=np.uint8)
    xs = [attrs['x'] for _, attrs in G.nodes(data=True)]
    ys = [attrs['y'] for _, attrs in G.nodes(data=True)]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    ### Get grid position of each node
    def to_grid(x, y):
        gx = int((x - min_x)/(max_x - min_x)*GRID_SIZE)
        gy = int((y - min_y)/(max_y - min_y)*GRID_SIZE)
        return gx, gy
        
    ### Find the number of lanes for each edge in the graph
    def get_lane_count(val, default=1):
        if val is None:
            return default
        if isinstance(val, (int, float)):
            return int(val)
        if isinstance(val, list):
            return sum([int(value) for value in val])
        if isinstance(val, str):
            return int(val)
            
    def intersection_radius(node, G):
        widths = []
        
        for u, v, attrs in G.in_edges(node, data=True):
            widths.append(get_lane_count(attrs.get('lanes', 1)))

        for u, v, attrs in G.out_edges(node, data=True):
            widths.append(get_lane_count(attrs.get('lanes', 1)))
            
        if not widths:
            return 1
            
        return max(1, max(widths) // 2)

    ### Set all roads and intersections in the discretized grid
    for u, v, attrs in G.edges(data=True):
        x1, y1 = G.nodes[u]['x'], G.nodes[u]['y']
        x2, y2 = G.nodes[v]['x'], G.nodes[v]['y']
        
        num_lanes = get_lane_count(attrs.get('lanes', 1))
        radius = max(1, num_lanes//2)
        
        gx1, gy1 = to_grid(x1, y1)
        gx2, gy2 = to_grid(x2, y2)
        
        rr, cc = line(gy1, gx1, gy2, gx2)
        prev_r, prev_c = None, None
        for r, c in zip(rr, cc):
            dr, dc = disk((r, c), radius, shape=grid.shape)
            grid[dr, dc] = 1
            
            if prev_r is not None:
                dr_step = r - prev_r
                dc_step = c - prev_c
                
                if abs(dr_step) == 1 and abs(dc_step) == 1:
                    rr_fix, cc_fix = disk((prev_r, c), radius, shape=grid.shape)
                    grid[rr_fix, cc_fix] = 1
                    
            prev_r, prev_c = r, c
            
    intersections = [n for n in G.nodes() if G.in_degree(n) + G.out_degree(n) >= 3]
    traffic_lights = {}
    for n in intersections:
        x, y = G.nodes[n]['x'], G.nodes[n]['y']
        gx, gy = to_grid(x, y)
        r = intersection_radius(n, G)
        traffic_lights[n] = TrafficLight(node=n, gx=gx, gy=gy, radius=r)
        rr, cc = disk((gy, gx), r, shape=grid.shape)
        grid[rr, cc] = 2
        
    return grid, traffic_lights
    
def render_scene(grid, traffic_lights, robots):
    h, w = grid.shape
    img = np.zeros((h,w,3), dtype=np.float32)
    
    img[grid == 1] = [0.6, 0.6, 0.6]
    img[grid == 2] = [1.0, 0.0, 0.0]
    
    for light in traffic_lights.values():
        rr, cc = disk((light.gy, light.gx), light.radius, shape=grid.shape)
        if light.green:
            img[rr, cc] = [0.0, 1.0, 0.0]
        else:
            img[rr, cc] = [1.0, 0.0, 0.0]
            
    for robot in robots:
        if not robot.finished:
            img[robot.pos[1], robot.pos[0]] = [0.0, 0.0, 1.0]

    return img
    
def run_simulation(grid, traffic_lights, robots):
    light_lookup = build_light_lookup(traffic_lights)
    reservation_table = {}

    with McapLogger("simulation.mcap") as mcap_logger:
        for step in range(NUM_STEPS):
            print(step)
            
            all_finished = True
            for robot in robots:
                if not robot.finished:
                    all_finished = False
                    
            if all_finished:
                break
            
            for light in traffic_lights.values():
                light.update()
                
            intersection_control = build_intersection_control_grid(grid, traffic_lights)
            #for robot in robots:
            #    if not robot.finished:
            #        robot.compute_path(reservation_table)
            
            for robot in robots:
                robot.intersection_control = intersection_control
                robot.step(step, reservation_table)
                
            mcap_logger.log_step(step, robots)
                
            signal_grid = np.zeros_like(grid)
            robot_grid = np.zeros_like(grid)
            
            for light in traffic_lights.values():
                rr, cc = disk((light.gy, light.gx), light.radius, shape=grid.shape)
                signal_grid[rr, cc] = 3 if light.green else 2
                
            for robot in robots:
                if not robot.finished:
                    robot_grid[robot.pos[1], robot.pos[0]] = 4
                    
            display_grid = render_scene(grid, traffic_lights, robots)
        
            plt.imshow(display_grid, origin="lower")
            plt.pause(0.01)
            plt.clf()
    

def main():
    x_min, x_max = -112.07209, -112.04741
    y_min, y_max = 33.46403, 33.49567
    if USE_FILE_DATA:
        reduced_G = ox.load_graphml("osm_graph.graphml")
    else:
        ### Build a graph of OpenStreetMap data for the city of Phoenix, Arizona
        print("Obtaining OpenStreetMap data...")
        G = fetch_map_data("Phoenix, AZ, USA")
        print("Truncating the data...")
        reduced_G = create_bounding_box(G, x_min, x_max, y_min, y_max)
        ox.save_graphml(reduced_G, filepath="osm_graph.graphml")
        
    ### Discretize the map and plot it using a 1000x1000 grid
    print("Converting the data to a discretized grid...")
    grid, traffic_lights = discretize_graph(reduced_G, x_min, x_max, y_min, y_max) 
    intersection_control = build_intersection_control_grid(grid, traffic_lights)       
    
    ### Build lights and robots
    print("Placing the robots on the grid")
    grid_graph = build_grid_graph(grid)
    starts_goals = create_robots(grid, NUM_AGENTS)
    robots = [Robot(start=start, goal=goal, grid=grid, grid_graph=grid_graph, intersection_control=intersection_control, robot_id=i) for i, (start, goal) in enumerate(starts_goals)]
    
    ### Run simulation
    print("Running simulation")
    run_simulation(grid, traffic_lights, robots)

if __name__=="__main__":
    main()
