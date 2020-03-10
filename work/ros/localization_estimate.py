#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import copy
import numpy as np
from scipy.stats import norm
import rospy
import time

# Robot motion commands:
# http://docs.ros.org/api/geometry_msgs/html/msg/Twist.html
from geometry_msgs.msg import Twist
# Laser scan message:
# http://docs.ros.org/api/sensor_msgs/html/msg/LaserScan.html
from sensor_msgs.msg import LaserScan
# For groundtruth information.
from gazebo_msgs.msg import ModelStates
from tf.transformations import euler_from_quaternion
# For displaying particles.
# http://docs.ros.org/api/sensor_msgs/html/msg/PointCloud.html
from sensor_msgs.msg import PointCloud
from sensor_msgs.msg import ChannelFloat32
from geometry_msgs.msg import Point32
from std_msgs.msg import Header
# Odometry.
from nav_msgs.msg import Odometry


# Constants used for indexing.
X = 0
Y = 1
YAW = 2

W_MAX = 2.84 #rad/s
U_MAX = 0.22 #m/s

ROBOT_RADIUS = 0.105 / 2.
WALL_OFFSET = 2.
CYLINDER_POSITION = np.array([.3, .2], dtype=np.float32)
CYLINDER_RADIUS = .3 + ROBOT_RADIUS

def sigmoid(x):
  return np.tanh(x/2)

def braitenberg(front, front_left, front_right, left, right):
  # u in [m/s]
  # w in [rad/s] going counter-clockwise.
  closestDistSide = 0.05
  closestDistFront = 0.4
  u = sigmoid(front-closestDistFront) * sigmoid(front_left+closestDistSide) * sigmoid(front_right+closestDistSide)
  w = sigmoid(5/front_right + 1/sigmoid(right) - 5/front_left - 1/sigmoid(left))
  return u, w

class Particle(object):
  """Represents a particle."""

  def __init__(self):
    self._pose = np.zeros(3, dtype=np.float32)
    self._weight = 1.
  
  def move(self, dt): 
    delp = np.zeros(3, dtype=np.float32)
    dt = 0.001

    u =  np.random.random_sample()*(U_MAX*2) - U_MAX
    w =  np.random.random_sample()*(W_MAX*2) - W_MAX

    delp[X] = u * dt
    delp[Y] += 0.
    delp[YAW] = w * dt

    self.augment(delp)


  def augment(self, delta_pose):
    
    
    forward_vel = np.abs(delta_pose[X])
    rot_vel = np.abs(delta_pose[YAW])
    world_delta_pose = delta_pose.copy()
    	    

    # Apply motion model
    if forward_vel > 0:
      world_delta_pose[X] = np.random.normal(world_delta_pose[X], forward_vel * self._weight)
      world_delta_pose[Y] = np.random.normal(world_delta_pose[Y], forward_vel * self._weight)
    if rot_vel > 0:
      world_delta_pose[YAW] = np.random.normal(world_delta_pose[YAW], rot_vel * self._weight)

    # Transform to world coords
    predicted_yaw = self._pose[YAW] + world_delta_pose[YAW] / 2 # Assume that this is a good enough estimate
    s, c = np.sin(predicted_yaw), np.cos(predicted_yaw)
    rot_mat = np.array(((c, -s), (s, c)))
    world_delta_pose[0:2] = np.matmul(rot_mat, delta_pose[0:2])

    self._pose += world_delta_pose

  def compute_weight(self, estloc, conf):
    prob = normal.pdf(self._pose, estloc, np.sqrt(conf))
    self._weight = prob  
    #self._weight = 1/prob 

  @property
  def pose(self):
    return self._pose

  @property
  def weight(self):
    return self._weight


class SimpleLaser(object):
  def __init__(self):
    rospy.Subscriber('/scan', LaserScan, self.callback)
    self._angles = [0., np.pi / 4., -np.pi / 4., np.pi / 2., -np.pi / 2.]
    self._width = np.pi / 180. * 3.1  # 3.1 degrees cone of view (3 rays).
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
    rospy.Subscriber('/gazebo/model_states', ModelStates, self.callback)
    self._pose = np.array([np.nan, np.nan, np.nan], dtype=np.float32)
    self._name = name

  def callback(self, msg):
    idx = [i for i, n in enumerate(msg.name) if n == self._name]
    if not idx:
      raise ValueError('Specified name "{}" does not exist.'.format(self._name))
    idx = idx[0]
    self._pose[X] = msg.pose[idx].position.x
    self._pose[Y] = msg.pose[idx].position.y
    _, _, yaw = euler_from_quaternion([
        msg.pose[idx].orientation.x,
        msg.pose[idx].orientation.y,
        msg.pose[idx].orientation.z,
        msg.pose[idx].orientation.w])
    self._pose[YAW] = yaw

  @property
  def ready(self):
    return not np.isnan(self._pose[0])

  @property
  def pose(self):
    return self._pose



def run(args):
  rospy.init_node('localization')

  # Update control every 100 ms.
  rate_limiter = rospy.Rate(100)
  publisher = rospy.Publisher('/cmd_vel', Twist, queue_size=5)
  particle_publisher = rospy.Publisher('/particles', PointCloud, queue_size=1)
  laser = SimpleLaser()
  # Keep track of groundtruth position for plotting purposes.
  groundtruth = GroundtruthPose()
  pose_history = []
  with open('/tmp/gazebo_exercise.txt', 'w'):
    pass

  num_particles = 50
  particles = [Particle() for _ in range(num_particles)]

  frame_id = 0
  while not rospy.is_shutdown():
    # Make sure all measurements are ready.
    if not laser.ready or not motion.ready or not groundtruth.ready:
      rate_limiter.sleep()
      continue

    #is the model changing
    u, w = braitenberg(*laser.measurements)
    vel_msg = Twist()
    vel_msg.linear.x = u
    vel_msg.angular.z = w
    publisher.publish(vel_msg)

    # Update particle positions and weights.
    total_weight = 0.
    for i, p in enumerate(particles):
      p.move()
      p.compute_weight(est_pos, var)
      total_weight += p.weight

    # Low variance re-sampling of particles.
    new_particles = []
    random_weight = np.random.rand() * total_weight / num_particles
    current_boundary = particles[0].weight
    j = 0
    for m in range(len(particles)):
      next_boundary = random_weight + m * total_weight / num_particles
      while next_boundary > current_boundary: 
        j = j + 1
        if j >= num_particles:
          j = num_particles - 1
        current_boundary = current_boundary + particles[j].weight
      new_particles.append(copy.deepcopy(particles[j]))
    particles = new_particles

    # Publish particles.
    particle_msg = PointCloud()
    particle_msg.header.seq = frame_id
    particle_msg.header.stamp = rospy.Time.now()
    particle_msg.header.frame_id = '/odom'
    intensity_channel = ChannelFloat32()
    intensity_channel.name = 'intensity'
    particle_msg.channels.append(intensity_channel)
    for p in particles:
      pt = Point32()
      pt.x = p.pose[X]
      pt.y = p.pose[Y]
      pt.z = .05
      particle_msg.points.append(pt)
      intensity_channel.values.append(p.weight)
    particle_publisher.publish(particle_msg)

    # Log groundtruth and estimated positions in /tmp/gazebo_exercise.txt
    poses = np.array([p.pose for p in particles], dtype=np.float32)
    median_pose = np.median(poses, axis=0)
    pose_history.append(np.concatenate([groundtruth.pose, median_pose], axis=0))
    if len(pose_history) % 10:
      with open('/tmp/gazebo_exercise.txt', 'a') as fp:
        fp.write('\n'.join(','.join(str(v) for v in p) for p in pose_history) + '\n')
        pose_history = []
    rate_limiter.sleep()
    frame_id += 1


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Runs a particle filter')
  args, unknown = parser.parse_known_args()
  try:
    run(args)
  except rospy.ROSInterruptException:
    pass
