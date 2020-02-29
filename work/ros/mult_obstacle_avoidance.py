#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import rospy

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


class SimpleLaser(object):
  def __init__(self, name):
    rospy.Subscriber('/' + name + '/scan', LaserScan, self.callback)
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

def run(args):
  avoidance_method = globals()[args.mode]
  name = args.name
  rospy.init_node(name + '_obstacle_avoidance')

  # Update control every 100 ms.
  rate_limiter = rospy.Rate(100)
  publisher = rospy.Publisher('/' + name + '/cmd_vel', Twist, queue_size=5)
  laser = SimpleLaser(name)

  while not rospy.is_shutdown():
    # Make sure all measurements are ready.
    if not laser.ready:
      rate_limiter.sleep()
      continue

    u, w = avoidance_method(*laser.measurements)
    vel_msg = Twist()
    vel_msg.linear.x = u
    vel_msg.angular.z = w
    publisher.publish(vel_msg)
    print("published to ", '/' + name + '/cmd_vel')
    rate_limiter.sleep()


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Runs obstacle avoidance')
  parser.add_argument('--mode', action='store', default='braitenberg', help='Method.', choices=['braitenberg', 'rule_based'])
  parser.add_argument('--name', action='store', default='turtlebot3_burger', help='Name.')
  args, unknown = parser.parse_known_args()
  try:
    run(args)
  except rospy.ROSInterruptException:
    pass
