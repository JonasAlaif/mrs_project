from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy.ndimage import gaussian_filter
import argparse
import matplotlib.pylab as plt
import numpy as np
import yaml
import os
import re
#import noise

# Constants used for indexing.
X = 0
Y = 1
YAW = 2

WALL_OFFSET = 5.
GOAL_POSITION = np.array([3, 3], dtype=np.float32)
START_POSITION = np.array([-1, -3], dtype=np.float32)
MAX_SPEED = .5

def sigmoid(x):
  return np.tanh(x)

def get_velocity_to_reach_goal(position, goal_position):
  v = np.zeros(2, dtype=np.float32)
  to_goal = goal_position - position
  to_goal_magnitude = np.linalg.norm(to_goal)
  if to_goal_magnitude < 1e-3:
    return 0
  # Normalise velocity to be <= MAX_SPEED
  v = MAX_SPEED * sigmoid(to_goal_magnitude * 3) * (to_goal / to_goal_magnitude)
  # Add perlin noise to avoid travelling in a straight line and getting stuck
  #perlin = np.array([noise.pnoise2(position[0], position[1]), noise.pnoise2(position[0], position[1], base=1)])
  #v = 2./3. * v + 1./3. * v * perlin
  return v


def get_velocity_to_avoid_obstacles(position, obstacle_map):
  v = np.zeros(2, dtype=np.float32)
  v = 10 * MAX_SPEED * obstacle_map.get_gradient(position)
  return v


def normalize(v):
  n = np.linalg.norm(v)
  if n < 1e-2:
    return np.zeros_like(v)
  return v / n


def cap(v, max_speed):
  n = np.linalg.norm(v)
  if n > max_speed:
    return v / n * max_speed
  return v


def get_velocity(position, obstacle_map, mode='all'):
  if mode in ('goal', 'all'):
    v_goal = get_velocity_to_reach_goal(position, GOAL_POSITION)
  else:
    v_goal = np.zeros(2, dtype=np.float32)
  if mode in ('obstacle', 'all'):
    v_avoid = get_velocity_to_avoid_obstacles(position, obstacle_map)
  else:
    v_avoid = np.zeros(2, dtype=np.float32)
  v = v_goal + v_avoid
  return cap(v, max_speed=MAX_SPEED)



# Defines an occupancy grid.
class ObstacleMap(object):
  def __init__(self, values, origin, resolution):
    self._original_values = values.copy()

    # Inflate obstacles (using a convolution).
    blurred_map = values.copy()
    blurred_map = gaussian_filter(blurred_map, sigma=7)
    blurred_map = np.multiply(blurred_map, values)
    blurred_map = gaussian_filter(blurred_map, sigma=3)
    blurred_map = np.multiply(blurred_map, values)
    self._blurred_map = gaussian_filter(blurred_map, sigma=1)
    gx, gy = np.gradient(self._blurred_map)
    self._values = np.stack([gx, gy], axis=-1)
    
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
                       self._origin[X] + self._original_values.shape[0] * self._resolution,
                       self._origin[Y],
                       self._origin[Y] + self._original_values.shape[1] * self._resolution])
    plt.set_cmap('gray')

  def draw_avoidance(self):
    plt.imshow(self._blurred_map.T, interpolation='none', origin='lower',
               extent=[self._origin[X],
                       self._origin[X] + self._blurred_map.shape[0] * self._resolution,
                       self._origin[Y],
                       self._origin[Y] + self._blurred_map.shape[1] * self._resolution])
    plt.set_cmap('gray')

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

  def get_gradient(self, position):
    return self._values[self.get_index(position)]


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


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Runs obstacle avoidance with a potential field')
  parser.add_argument('--mode', action='store', default='all', help='Which velocity field to plot.', choices=['obstacle', 'goal', 'all'])
  parser.add_argument('--map', action='store', default='map_city', help='Which map to use.')
  args, unknown = parser.parse_known_args()

  # Load map.
  with open(args.map + '.yaml') as fp:
    data = yaml.load(fp)
  img = read_pgm(os.path.join(os.path.dirname(args.map), data['image']))
  obstacle_map = ObstacleMap(img, data['origin'], data['resolution'])

  # Plot environment.
  fig, ax = plt.subplots()
  obstacle_map.draw()
  #obstacle_map.draw_avoidance()
  
  plt.scatter(START_POSITION[X], START_POSITION[Y], s=10, marker='o', color='green', zorder=1000)
  plt.scatter(GOAL_POSITION[X], GOAL_POSITION[Y], s=10, marker='o', color='red', zorder=1000)

  # Plot field.
  Xs, Ys = np.meshgrid(np.linspace(-WALL_OFFSET, WALL_OFFSET, 30),
                     np.linspace(-WALL_OFFSET, WALL_OFFSET, 30))
  U = np.zeros_like(Xs)
  V = np.zeros_like(Xs)
  for i in range(len(Xs)):
    for j in range(len(Xs[0])):
      velocity = get_velocity(np.array([Xs[i, j], Ys[i, j]]), obstacle_map, args.mode)
      U[i, j] = velocity[0]
      V[i, j] = velocity[1]
  plt.quiver(Xs, Ys, U, V, units='width')

  # Plot a simple trajectory from the start position.
  # Uses Euler integration.
  dt = 0.01
  x = START_POSITION
  positions = [x]
  for t in np.arange(0., 40., dt):
    v = get_velocity(x, obstacle_map, args.mode)
    x = x + v * dt
    positions.append(x)
  positions = np.array(positions)
  plt.plot(positions[:, 0], positions[:, 1], lw=2, c='r')

  plt.axis('equal')
  plt.xlabel('x')
  plt.ylabel('y')
  plt.xlim([-.5 - WALL_OFFSET, WALL_OFFSET + .5])
  plt.ylim([-.5 - WALL_OFFSET, WALL_OFFSET + .5])
  plt.show()