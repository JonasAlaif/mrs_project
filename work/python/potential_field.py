from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import matplotlib.pylab as plt
import numpy as np
#import noise

WALL_OFFSET = 2.
#CYLINDER_POSITION = np.array([.3, .2], dtype=np.float32)
CYLINDER_POSITION = np.array([[.5, .0], [.0, .5]], dtype=np.float32)
CYLINDER_RADIUS = [.3, .3]
GOAL_POSITION = np.array([1.5, 1.5], dtype=np.float32)
START_POSITION = np.array([-1.5, -1.5], dtype=np.float32)
MAX_SPEED = .5
MAX_SPEED_SQRT = .7

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


def get_velocity_to_avoid_obstacles(position, obstacle_positions, obstacle_radii):
  v = np.zeros(2, dtype=np.float32)
  for (ob_pos, ob_radius) in zip(obstacle_positions, obstacle_radii):
    from_obst = position - ob_pos
    from_obst_magnitude = np.linalg.norm(from_obst)
    # Rotational force, to avoid getting stuck
    from_obst_left = np.array((-from_obst[1], from_obst[0]))
    # Calculate normalised vector away from obstacle, including rotational force in the second case
    #from_obst_norm = from_obst / (from_obst_magnitude + 1e-6)
    from_obst_norm = (from_obst + from_obst_left * .05) / (from_obst_magnitude * 1.05 + 1e-6)
    if from_obst_magnitude <= ob_radius:
      v += from_obst_norm * MAX_SPEED
    else:
      v += MAX_SPEED * from_obst_norm / np.power(from_obst_magnitude - ob_radius + 1, 2)
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


def get_velocity(position, mode='all'):
  if mode in ('goal', 'all'):
    v_goal = get_velocity_to_reach_goal(position, GOAL_POSITION)
  else:
    v_goal = np.zeros(2, dtype=np.float32)
  if mode in ('obstacle', 'all'):
    v_avoid = get_velocity_to_avoid_obstacles(
      position,
      CYLINDER_POSITION,
      CYLINDER_RADIUS)
  else:
    v_avoid = np.zeros(2, dtype=np.float32)
  v = v_goal + v_avoid
  return cap(v, max_speed=MAX_SPEED)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Runs obstacle avoidance with a potential field')
  parser.add_argument('--mode', action='store', default='all', help='Which velocity field to plot.', choices=['obstacle', 'goal', 'all'])
  args, unknown = parser.parse_known_args()

  fig, ax = plt.subplots()
  # Plot field.
  X, Y = np.meshgrid(np.linspace(-WALL_OFFSET, WALL_OFFSET, 30),
                     np.linspace(-WALL_OFFSET, WALL_OFFSET, 30))
  U = np.zeros_like(X)
  V = np.zeros_like(X)
  for i in range(len(X)):
    for j in range(len(X[0])):
      velocity = get_velocity(np.array([X[i, j], Y[i, j]]), args.mode)
      U[i, j] = velocity[0]
      V[i, j] = velocity[1]
  plt.quiver(X, Y, U, V, units='width')

  # Plot environment.
  for i in range(len(CYLINDER_RADIUS)):
    ax.add_artist(plt.Circle(CYLINDER_POSITION[i], CYLINDER_RADIUS[i], color='gray'))
  plt.plot([-WALL_OFFSET, WALL_OFFSET], [-WALL_OFFSET, -WALL_OFFSET], 'k')
  plt.plot([-WALL_OFFSET, WALL_OFFSET], [WALL_OFFSET, WALL_OFFSET], 'k')
  plt.plot([-WALL_OFFSET, -WALL_OFFSET], [-WALL_OFFSET, WALL_OFFSET], 'k')
  plt.plot([WALL_OFFSET, WALL_OFFSET], [-WALL_OFFSET, WALL_OFFSET], 'k')

  # Plot a simple trajectory from the start position.
  # Uses Euler integration.
  dt = 0.01
  x = START_POSITION
  positions = [x]
  for t in np.arange(0., 20., dt):
    v = get_velocity(x, args.mode)
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