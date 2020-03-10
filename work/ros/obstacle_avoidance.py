#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import rospy
import sys

# Robot motion commands:
# http://docs.ros.org/api/geometry_msgs/html/msg/Twist.html
from geometry_msgs.msg import Twist
# Laser scan message:
# http://docs.ros.org/api/sensor_msgs/html/msg/LaserScan.html
from sensor_msgs.msg import LaserScan
# For groundtruth information.
from gazebo_msgs.msg import ModelStates
from tf.transformations import euler_from_quaternion

def sigmoid(x):
  return np.tanh(x/2)

def braitenberg(front, front_left, front_right, left, right):
  # u in [m/s]
  # w in [rad/s] going counter-clockwise.
  closestDistSide = 0.05
  closestDistFront = 0.4
  u = sigmoid(front-closestDistFront) * sigmoid(front_left+closestDistSide) * sigmoid(front_right+closestDistSide) * 3
  w = sigmoid(5/front_right + 1/sigmoid(right) - 5/front_left - 1/sigmoid(left))
  return u, w


def rule_based(front, front_left, front_right, left, right):
  u = 0.  # [m/s]
  w = 0.  # [rad/s] going counter-clockwise.

  if front < 0.4 or front_right < 0.2 or front_left < 0.2:
    # Hit wall, turn left to drive with wall on right
    u = -0.2
    w = 0.6
  elif front < 0.8:
    u = 0.2
    w = 0.3
  elif front_right > right * 1.6:
    # Turned to far away from wall, keep perpendicular to wall
    u = 0.3
    w = -0.4
  elif right < 0.7 and front_right < right * 1.1:
    # Driving towards wall, keep perpendicular
    u = 0.2
    w = 0.2
  elif right < 0.3:
    # Driving too close to wall, back off
    u = 0.15
    w = 0.3
  elif right < 0.7 and front_right > right * 1.5:
    # Driving away wall, keep perpendicular
    u = 0.3
    w = -0.2
  elif right < 0.7:
    u = 0.5
  else:
    u = 0.8

  return u, w

def check_line_of_sight(from_pos, to_pos, obstacle_map):
  uncertainty = 1
  curr_pos = from_pos.copy()
  goal_pos = to_pos.copy()
  step = to_pos - from_pos
  step = step / np.amax(np.abs(step)) * obstacle_map.resolution
  step_size = np.linalg.norm(step)
  while step_size < np.linalg.norm(goal_pos - curr_pos):
    curr_uncertainty = obstacle_map.get_visibility(curr_pos)
    uncertainty *= np.power(curr_uncertainty, step_size)
    #print(uncertainty)
    if uncertainty < 1e-10:
      return 0.0
    curr_pos += step
  return uncertainty

class SimpleLaser(object):
  def __init__(self, name):
    if name is not None:
      rospy.Subscriber('/' + name + '/scan', LaserScan, self.callback)
    else:
      rospy.Subscriber('/scan', LaserScan, self.callback)
    self._angles = [0., np.pi / 4., -np.pi / 4., np.pi / 2., -np.pi / 2.]
    self._width = np.pi / 180. * 10.  # 10 degrees cone of view.
    self._measurements = [float('inf')] * len(self._angles)
    self._indices = None

  def callback(self, msg):
    # Helper for angles.
    def _within(x, a, b):
      pi2 = np.pi * 2.
      x %= pi2
      a %= pi2
      b %= pi2
      if a < b:
        return a <= x and x <= b
      return a <= x or x <= b

    # Compute indices the first time.
    if self._indices is None:
      self._indices = [[] for _ in range(len(self._angles))]
      for i, d in enumerate(msg.ranges):
        angle = msg.angle_min + i * msg.angle_increment
        for j, center_angle in enumerate(self._angles):
          if _within(angle, center_angle - self._width / 2., center_angle + self._width / 2.):
            self._indices[j].append(i)

    ranges = np.array(msg.ranges)
    for i, idx in enumerate(self._indices):
      # We do not take the minimum range of the cone but the 10-th percentile for robustness.
      self._measurements[i] = np.percentile(ranges[idx], 10)

  @property
  def ready(self):
    return not np.isnan(self._measurements[0])

  @property
  def measurements(self):
    return self._measurements


class GroundtruthPose(object):
  def __init__(self, name='turtlebot3_burger'):
    self._subscriber = rospy.Subscriber('/gazebo/model_states', ModelStates, self.callback)
    self._pose = np.array([np.nan, np.nan, np.nan], dtype=np.float32)
    self._name = name

  def callback(self, msg):
    idx = [i for i, n in enumerate(msg.name) if n == self._name]
    if not idx:
      self._subscriber.unregister()
      #raise ValueError('Specified name "{}" does not exist.'.format(self._name))
      print('Specified name "{}" does not exist.'.format(self._name), file=sys.stderr)
    else:
      idx = idx[0]
      self._pose[0] = msg.pose[idx].position.x
      self._pose[1] = msg.pose[idx].position.y
      _, _, yaw = euler_from_quaternion([
          msg.pose[idx].orientation.x,
          msg.pose[idx].orientation.y,
          msg.pose[idx].orientation.z,
          msg.pose[idx].orientation.w])
      self._pose[2] = yaw

  @property
  def ready(self):
    return not np.isnan(self._pose[0])

  @property
  def pose(self):
    return self._pose

  def pose_with_uncertainty(self, observer_poses, obstacle_map):
    uncertainties = np.array([check_line_of_sight(observer_pose[:2], self.pose[:2], obstacle_map) for observer_pose in observer_poses])
    if len(uncertainties) == 0:
      return (self.pose, 0)
    raw_uncertainty = np.amax(uncertainties) # 1 for certain, 0 for no knowledge
    return (self.pose, raw_uncertainty)


  def observed_pose(self, observer_poses, obstacle_map):
    true_pose, uncertainty = self.pose_with_uncertainty(observer_poses, obstacle_map)
    if uncertainty < 1e-3:
      return (np.zeros_like(true_pose), float('inf'))
    if uncertainty > 1 - 1e-3:
      return (true_pose, 0.0)
    new_pose = true_pose.copy()
    variance = 1.0 / uncertainty - 1.0
    new_pose[:2] = np.random.normal(new_pose[:2], variance)
    #print("TP: ", true_pose, ", resampled to: ", new_pose)
    return (new_pose, variance)
  

def run(args):
  rospy.init_node('obstacle_avoidance')
  avoidance_method = globals()[args.mode]

  # Update control every 100 ms.
  rate_limiter = rospy.Rate(100)
  publisher = rospy.Publisher('/cmd_vel', Twist, queue_size=5)
  laser = SimpleLaser(None)
  # Keep track of groundtruth position for plotting purposes.
  groundtruth = GroundtruthPose()
  pose_history = []
  with open('/tmp/gazebo_exercise.txt', 'w'):
    pass

  while not rospy.is_shutdown():
    # Make sure all measurements are ready.
    if not laser.ready or not groundtruth.ready:
      rate_limiter.sleep()
      continue

    u, w = avoidance_method(*laser.measurements)
    vel_msg = Twist()
    vel_msg.linear.x = u
    vel_msg.angular.z = w
    publisher.publish(vel_msg)

    # Log groundtruth positions in /tmp/gazebo_exercise.txt
    pose_history.append(groundtruth.pose)
    if len(pose_history) % 10:
      with open('/tmp/gazebo_exercise.txt', 'a') as fp:
        fp.write('\n'.join(','.join(str(v) for v in p) for p in pose_history) + '\n')
        pose_history = []
    rate_limiter.sleep()


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Runs obstacle avoidance')
  parser.add_argument('--mode', action='store', default='braitenberg', help='Method.', choices=['braitenberg', 'rule_based'])
  args, unknown = parser.parse_known_args()
  try:
    run(args)
  except rospy.ROSInterruptException:
    pass
