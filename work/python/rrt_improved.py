from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import argparse
import matplotlib.pylab as plt
import matplotlib.patches as patches
import numpy as np
import os
import re
import scipy.signal
import time
import yaml

import multiprocessing
import threading


# Constants used for indexing.
X = 0
Y = 1
YAW = 2

# Constants for occupancy grid.
FREE = 0
UNKNOWN = 1
OCCUPIED = 2

ROBOT_RADIUS = 0.105 / 2.
GOAL_POSITION = np.array([-7.5, -7.4], dtype=np.float32)  # Any orientation is good.
START_POSE = np.array([6.5, 7.5, 0], dtype=np.float32)
MAX_ITERATIONS = 1000

printlock = multiprocessing.Lock()

def sample_random_position(occupancy_grid):
  position = np.ones(2, dtype=np.float32)
  # Get indicies of all free positions
  free_positions = np.argwhere(occupancy_grid.values == FREE)
  # Choose a random index
  free_position = free_positions[np.random.choice(free_positions.shape[0])]
  # Get the world position of that point
  position = occupancy_grid.get_position(free_position[0], free_position[1])
  return position


def adjust_pose(node, final_position, occupancy_grid):
  final_pose = node.pose.copy()
  final_pose[:2] = final_position
  final_node = Node(final_pose)
  # YAW that the robot would be in if it reached the final_position
  angle_pose_to_final = np.arctan2(final_node.pose[Y] - node.pose[Y], final_node.pose[X] - node.pose[X])
  final_node.pose[YAW] = angle_pose_to_final - (node.pose[YAW] - angle_pose_to_final)
  # Get the arc connecting the two points
  (center, radius) = find_circle(node, final_node)
  start_angle = np.arctan2(node.pose[Y] - center[Y], node.pose[X] - center[X])
  end_angle = np.arctan2(final_node.pose[Y] - center[Y], final_node.pose[X] - center[X])
  # Choose angle step to be small enough that we don't miss any points on arc
  angle_step = ROBOT_RADIUS / radius * .9
  if end_angle < start_angle:
    angle_step = -angle_step
  # Get list of all angles we must check on circle border
  angle_samples = np.arange(start_angle + 2*angle_step, end_angle - 2*angle_step, angle_step)
  position_samples = [(center + [np.cos(angle) * radius, np.sin(angle) * radius]) for angle in angle_samples]
  for position in position_samples:
    if sample_position(position, occupancy_grid) != 0:
      return None
  return final_node

def sample_position(position, occupancy_grid):
  og_position = occupancy_grid.get_index(position)
  cell_count = np.ceil(ROBOT_RADIUS / occupancy_grid.resolution).astype(int)
  return_sum = 0
  # Approximate robot to a square, check all positions within square
  for x in range(-cell_count + 1, cell_count):
    for y in range(-cell_count + 1, cell_count):
      return_sum += occupancy_grid.values[og_position[X] + x, og_position[Y] + y]
  return return_sum


# Defines an occupancy grid.
class OccupancyGrid(object):
  def __init__(self, values, origin, resolution):
    self._original_values = values.copy()
    self._values = values.copy()
    # Inflate obstacles (using a convolution).
    inflated_grid = np.zeros_like(values)
    inflated_grid[values == OCCUPIED] = 1.
    w = 2 * int(ROBOT_RADIUS / resolution) + 1
    inflated_grid = scipy.signal.convolve2d(inflated_grid, np.ones((w, w)), mode='same')
    self._values[inflated_grid > 0.] = OCCUPIED
    self._origin = np.array(origin[:2], dtype=np.float32)
    self._origin -= resolution / 2.
    assert origin[YAW] == 0.
    self._resolution = resolution

  @property
  def values(self):
    return self._values

  @property
  def resolution(self):
    return self._resolution

  @property
  def origin(self):
    return self._origin

  def draw(self):
    plt.imshow(self._original_values.T, interpolation='none', origin='lower',
               extent=[self._origin[X],
                       self._origin[X] + self._values.shape[0] * self._resolution,
                       self._origin[Y],
                       self._origin[Y] + self._values.shape[1] * self._resolution])
    plt.set_cmap('gray_r')

  def get_index(self, position):
    idx = ((position - self._origin) / self._resolution).astype(np.int32)
    if len(idx.shape) == 2:
      idx[:, 0] = np.clip(idx[:, 0], 0, self._values.shape[0] - 1)
      idx[:, 1] = np.clip(idx[:, 1], 0, self._values.shape[1] - 1)
      return (idx[:, 0], idx[:, 1])
    idx[0] = np.clip(idx[0], 0, self._values.shape[0] - 1)
    idx[1] = np.clip(idx[1], 0, self._values.shape[1] - 1)
    return tuple(idx)

  def get_position(self, i, j):
    return np.array([i, j], dtype=np.float32) * self._resolution + self._origin

  def is_occupied(self, position):
    return self._values[self.get_index(position)] == OCCUPIED

  def is_free(self, position):
    return self._values[self.get_index(position)] == FREE


# Defines a node of the graph.
class Node(object):
  def __init__(self, pose):
    self._pose = pose.copy()
    self._neighbors = []
    self._parent = None
    self._cost = 0.

  @property
  def pose(self):
    return self._pose

  def add_neighbor(self, node):
    self._neighbors.append(node)

  @property
  def parent(self):
    return self._parent

  @parent.setter
  def parent(self, node):
    self._parent = node

  @property
  def neighbors(self):
    return self._neighbors

  @property
  def position(self):
    return self._pose[:2]

  @property
  def yaw(self):
    return self._pose[YAW]

  @property
  def direction(self):
    return np.array([np.cos(self._pose[YAW]), np.sin(self._pose[YAW])], dtype=np.float32)

  @property
  def cost(self):
      return self._cost

  @cost.setter
  def cost(self, c):
    self._cost = c

def calc_cost(parent, child_pos, police):
  dist = np.linalg.norm(parent.position - child_pos)
  police_distance = float('inf')
  for pol in police:
    # TODO rewrite this (currently just copy-pasted from https://gist.github.com/nim65s/5e9902cd67f094ce65b0)
    pol_d = seg_dist(parent.position, child_pos, pol[0])
    if pol_d < police_distance:
      police_distance = pol_d
    #print('pol_dist: ', police_distance)
  #police_distance = 0 # TODO calculate
  # don't actually know if this works properly
  return parent.cost + dist + 10/(police_distance)

# TODO rewrite this (currently just copy-pasted from https://gist.github.com/nim65s/5e9902cd67f094ce65b0)
def seg_dist(A, B, P):
  """ segment line AB, point P, where each one is an array([x, y]) """
  if all(A == P) or all(B == P):
    return 0
  if np.arccos(np.dot((P - A) / np.linalg.norm(P - A), (B - A) / np.linalg.norm(B - A))) > np.pi / 2:
    return np.linalg.norm(P - A)
  if np.arccos(np.dot((P - B) / np.linalg.norm(P - B), (A - B) / np.linalg.norm(A - B))) > np.pi / 2:
    return np.linalg.norm(P - B)
  return np.linalg.norm(np.cross(A-B, A-P))/np.linalg.norm(B-A)

def rrt_nocircle(start_pose, goal_position, occupancy_grid, police, num_iterations=MAX_ITERATIONS):
  # RRT builds a graph one node at a time.
  graph = []
  start_node = Node(start_pose)
  final_nodes = []
  if not occupancy_grid.is_free(goal_position[:2]):
    print('Goal position is not in the free space.')
    return start_node, None
  graph.append(start_node)
  for _ in range(num_iterations):
    # With a random chance, draw the goal position.
    if np.random.rand() < .05:
      position = goal_position[:2]
    else:
      position = sample_random_position(occupancy_grid)
    # Find closest (with respect to cost) node in graph.
    # In practice, one uses an efficient spatial structure (e.g., quadtree).
    graph_dists = [(n, np.linalg.norm(position - n.position)) for n in graph]
    potential_parents = filter(lambda (n, d): parent_filter_noyaw(position, (n, d)), graph_dists)
    min_cost = float('inf')
    u = None
    for (n, d) in potential_parents:
      cost = calc_cost(n, position, police)
      if min_cost > cost:
        # For the cost I use euclidean distance. It would be better to use the arc distance,
        # but then we would have to calulate arcs for each neighbour, which is more expensive
        min_cost = cost
        u = n
    if u == None:
      #print('u is None')
      continue
    if check_line_of_sight(u.pose[:2], position[:2], occupancy_grid):
      v = Node(np.append(position, 0))
    else:
      continue
    v.cost = min_cost
    u.add_neighbor(v)
    v.parent = u
    graph.append(v)
    if np.linalg.norm(v.position - goal_position[:2]) < .2:
      final_nodes.append(v)
      # Uncomment 'break' for reduced optimality but faster execution
      # Could potentially do a check that:
      # if we are not finding a better solution for a while, then break
      #break
  if len(final_nodes) == 0:
    return start_node, None
  return start_node, min(final_nodes, key=lambda final_node: final_node.cost)

def check_line_of_sight(from_pos, to_pos, obstacle_map):
  uncertainty = 1
  curr_pos = from_pos.copy()
  step = to_pos - from_pos
  step = step / np.amax(np.abs(step)) * obstacle_map.resolution
  step_size = np.linalg.norm(step)
  hasCollided = False
  while step_size < np.linalg.norm(to_pos - curr_pos):
    hasCollided = not obstacle_map.is_free(curr_pos)
    if hasCollided:
      #print('collision')
      break
    else:
      curr_pos += step 
  return not hasCollided

def rrt(start_pose, goal_position, occupancy_grid, num_iterations=MAX_ITERATIONS):
  # RRT builds a graph one node at a time.
  graph = []
  start_node = Node(start_pose)
  final_nodes = []
  if not occupancy_grid.is_free(goal_position):
    print('Goal position is not in the free space.')
    return start_node, None
  graph.append(start_node)
  for _ in range(num_iterations):
    # With a random chance, draw the goal position.
    if np.random.rand() < .05:
      position = goal_position
    else:
      position = sample_random_position(occupancy_grid)
    # Find closest (with respect to cost) node in graph.
    # In practice, one uses an efficient spatial structure (e.g., quadtree).
    graph_dists = [(n, np.linalg.norm(position - n.position)) for n in graph]
    potential_parents = filter(lambda (n, d): parent_filter(position, (n, d)), graph_dists)
    min_cost = float('inf')
    u = None
    for (n, d) in potential_parents:
      if min_cost > n.cost + d:
        # For the cost I use euclidean distance. It would be better to use the arc distance,
        # but then we would have to calulate arcs for each neighbour, which is more expensive
        min_cost = n.cost + d
        u = n
    if u == None:
      continue
    v = adjust_pose(u, position, occupancy_grid)
    if v is None:
      continue
    v.cost = min_cost
    u.add_neighbor(v)
    v.parent = u
    graph.append(v)
    if np.linalg.norm(v.position - goal_position) < .2:
      final_nodes.append(v)
      # Uncomment 'break' for reduced optimality but faster execution
      # Could potentially do a check that:
      # if we are not finding a better solution for a while, then break
      #break
  if len(final_nodes) == 0:
    return start_node, None
  return start_node, min(final_nodes, key=lambda final_node: final_node.cost)

def parent_filter(position, (n, d)):
  return d > .2 and d < 1.5 and n.direction.dot(position - n.position) / d > 0.70710678118

def parent_filter_noyaw(position, (n, d)):
  return d > .2 and d < 4.

def find_circle(node_a, node_b):
  def perpendicular(v):
    w = np.empty_like(v)
    w[X] = -v[Y]
    w[Y] = v[X]
    return w
  db = perpendicular(node_b.direction)
  dp = node_a.position - node_b.position
  t = np.dot(node_a.direction, db)
  if np.abs(t) < 1e-3:
    # By construction node_a and node_b should be far enough apart,
    # so they must be on opposite end of the circle.
    center = (node_b.position + node_a.position) / 2.
    radius = np.linalg.norm(center - node_b.position)
  else:
    radius = np.dot(node_a.direction, dp) / t
    center = radius * db + node_b.position
  return center, np.abs(radius)


def read_pgm(filename, byteorder='>'):
  """Read PGM file."""
  with open(filename, 'rb') as fp:
    buf = fp.read()
  try:
    header, width, height, maxval = re.search(
        b'(^P5\s(?:\s*#.*[\r\n])*'
        b'(\d+)\s(?:\s*#.*[\r\n])*'
        b'(\d+)\s(?:\s*#.*[\r\n])*'
        b'(\d+)\s(?:\s*#.*[\r\n]\s)*)', buf).groups()
  except AttributeError:
    raise ValueError('Invalid PGM file: "{}"'.format(filename))
  maxval = int(maxval)
  height = int(height)
  width = int(width)
  img = np.frombuffer(buf,
                      dtype='u1' if maxval < 256 else byteorder + 'u2',
                      count=width * height,
                      offset=len(header)).reshape((height, width))
  return img.astype(np.float32) / 255.


def draw_solution(start_node, police, final_node=None):
  ax = plt.gca()

  def draw_path(u, v, arrow_length=.1, color=(.8, .8, .8), lw=1):
    #du = u.direction
    #plt.arrow(u.pose[X], u.pose[Y], du[0] * arrow_length, du[1] * arrow_length,
    #          head_width=.05, head_length=.1, fc=color, ec=color)
    #dv = v.direction
    #plt.arrow(v.pose[X], v.pose[Y], dv[0] * arrow_length, dv[1] * arrow_length,
    #          head_width=.05, head_length=.1, fc=color, ec=color)
    #center, radius = find_circle(u, v)
    #du = u.position - center
    #theta1 = np.arctan2(du[1], du[0])
    #dv = v.position - center
    #theta2 = np.arctan2(dv[1], dv[0])
    ## Check if the arc goes clockwise.
    #if np.cross(u.direction, du).item() > 0.:
    #  theta1, theta2 = theta2, theta1
    #ax.add_patch(patches.Arc(center, radius * 2., radius * 2.,
    #                         theta1=theta1 / np.pi * 180., theta2=theta2 / np.pi * 180.,
    #                         color=color, lw=lw))
    plt.arrow(u.pose[X], u.pose[Y], v.pose[X] - u.pose[X], v.pose[Y] - u.pose[Y], head_width = 0.05, head_length = 0.1, fc=color, ec=color)

  points = []
  s = [(start_node, None)]  # (node, parent).
  while s:
    v, u = s.pop()
    if hasattr(v, 'visited'):
      continue
    v.visited = True
    # Draw path from u to v.
    if u is not None:
      draw_path(u, v)
    points.append(v.pose[:2])
    for w in v.neighbors:
      s.append((w, v))

  for p in police:
    plt.scatter(p[0][0], p[0][1], s=10, marker='o', color=[1, 0, 0])

  points = np.array(points)
  plt.scatter(points[:, 0], points[:, 1], s=10, marker='o', color=(.8, .8, .8))
  if final_node is not None:
    plt.scatter(final_node.position[0], final_node.position[1], s=10, marker='o', color='k')
    # Draw final path.
    v = final_node
    while v.parent is not None:
      draw_path(v.parent, v, color='k', lw=2)
      v = v.parent

def run_thread(values, times, i, num_i, j, num_j, start, end, occ, numiter):
  #print("start")
  time_start = time.time() * 1000
  start_node, end_node = rrt_nocircle(start, end, occ, num_iterations = numiter)
  time_end = time.time() * 1000
  #print("end")
  #values[j][i] = end_node is None
  printlock.acquire()
  print(j, ',', i, 'time:', time_end - time_start, 'ms', 'finished:', end_node is not None)
  printlock.release()
  return end_node is None, time_end - time_start


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Uses RRT to reach the goal.')
  parser.add_argument('--map', action='store', default='map_city_3', help='Which map to use.')
  args, unknown = parser.parse_known_args()

  # Load map.
  with open(args.map + '.yaml') as fp:
    data = yaml.load(fp)
  img = read_pgm(os.path.join(os.path.dirname(args.map), data['image']))
  occupancy_grid = np.empty_like(img, dtype=np.int8)
  occupancy_grid[:] = UNKNOWN
  occupancy_grid[img < .1] = OCCUPIED
  occupancy_grid[img > .9] = FREE
  # Transpose (undo ROS processing).
  occupancy_grid = occupancy_grid.T
  # Invert Y-axis.
  occupancy_grid = occupancy_grid[:, ::-1]
  occupancy_grid = OccupancyGrid(occupancy_grid, data['origin'], data['resolution'])

  start_node, final_node = rrt_nocircle(START_POSE, GOAL_POSITION, occupancy_grid)

  # Run RRT.
  num_j = 7
  num_i = 100
  iterations = 500
  #times = multiprocessing.Array('d', num_i * num_j)
  #values = multiprocessing.Array('d', num_i * num_j * 2)
  times = np.zeros((num_i, num_j))
  values = np.zeros((num_i, num_j, 2))
  p = multiprocessing.Pool(6)
  for j in range(num_j):
    for i in range(num_i):
      p.apply_async(func=run_thread, args=(values, times, i, num_i, j, num_j, START_POSE, GOAL_POSITION, occupancy_grid, iterations))
    iterations += 500
  
  p.close()
  p.join()

  #print(values)
  #print(times)


  for j in range(num_j):
    counter = sum(values)
    time_total = sum(times)
    #print('counter:', counter, 'ave time:', time_total/num_i)
  


  ## Plot environment.
  #fig, ax = plt.subplots()
  #occupancy_grid.draw()
  #plt.scatter(.3, .2, s=10, marker='o', color='green', zorder=1000)
  #draw_solution(start_node, final_node)
  #plt.scatter(START_POSE[0], START_POSE[1], s=10, marker='o', color='green', zorder=1000)
  #plt.scatter(GOAL_POSITION[0], GOAL_POSITION[1], s=10, marker='o', color='red', zorder=1000)

  #plt.axis('equal')
  #plt.xlabel('x')
  #plt.ylabel('y')
  #plt.xlim([-.5 - 2., 2. + .5])
  #plt.ylim([-.5 - 2., 2. + .5])
  #plt.show()

