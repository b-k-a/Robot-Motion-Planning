# Robot Grid Routing Simulation

This project simulates many robots driving through a discretized road network with traffic lights and computes routes for them on a grid.

## Overview

The code loads a road network (from OpenStreetMap or a saved file), converts it into a 2D grid, places robots and traffic lights, and then runs a time‑stepped simulation where robots move toward their goals while obeying signals. Grid cells represent drivable road surfaces and intersections; robots move one grid cell per step.

## Map and grid generation

- The script fetches or loads a directed road graph for a bounded geographic area using OSMnx.
- `create_bounding_box` trims the original road graph to a specified latitude/longitude rectangle.
- `discretize_graph` converts this graph into a \((GRID\_SIZE+1) \times (GRID\_SIZE+1)\) NumPy array. 
- `build_grid_graph` then builds an undirected NetworkX graph where each node is a grid coordinate `(x, y)` corresponding to a road or intersection cell, and edges connect 4‑connected neighbors with a uniform weight of 1.

## Traffic lights

- For each intersection node in the original road graph, a `TrafficLight` is created and mapped into grid coordinates. 
- A traffic light has:
  - `green` (whether the light is currently green),
  - `green_time` (how long a phase lasts, randomly chosen between `MIN_GREEN` and `MAX_GREEN`),
  - `timer` (how long it has been in the current phase), and
  - `radius` (how far around the node the light controls).  
- `build_intersection_control_grid` produces a grid of object references pointing to the controlling traffic light for each relevant cell.
- On each simulation step, `TrafficLight.update` advances the timer and toggles `green` when the phase duration elapses.

## Robots and simulation loop

- `create_robots` samples random road cells for each robot’s start and goal positions.
- Each `Robot` stores:
  - `pos`: current grid coordinate `(x, y)`,
  - `goal`: target grid coordinate,
  - `grid`: the map grid,
  - `grid_graph`: the NetworkX graph over road cells,
  - `intersection_control`: grid of traffic light controllers,
  - `stick_prob`: probability to randomly “stick” and skip movement in a time step,
  - `path`: the current planned route as a list of grid coordinates,
  - `finished`: flag for when the robot reaches its goal,
  - `robot_id`: integer identifier.

The main loop in `run_simulation` does, for each time step:

- Update every traffic light.
- Rebuild the `intersection_control` grid so robots see current green/red states.
- For each robot, call `robot.step()` to optionally recompute its path and move one cell.
- Render a visualization where roads, intersections, lights, and robots are colored appropriately, then show the frame with Matplotlib.

## Routing algorithm

Routing is handled by the `Robot.compute_path` method:

- Robots look within a fixed radius to see if other robots are surrounding them within that radius
- If there are no robots, action is determined by Dijkstra's shortest path algorithm
- If there are robots, it uses a game theory approach to determine the individual best response
- This algorithm is based on my published work "Situational Awareness for Safe and Robust Multi-Agent Interactions Under Uncertainty"
