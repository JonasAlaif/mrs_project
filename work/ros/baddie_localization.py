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

W_MAX = 2.84 #rad/s
U_MAX = 0.22 #m/s

# Constants used for indexing.
X = 0
Y = 1
YAW = 2

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


class Particle(object):
  """Represents a particle."""

  def __init__(self):
    self._pose = None
    self._weight = 1.
    self._ready = False

  @property
  def ready(self):
    return self._ready

  @property
  def pose(self):
    return self._pose

  @property
  def weight(self):
    return self._weight

  def initialize(self, start_pose):
    self._pose = start_pose.copy()
    self._ready = True

  def is_valid(self, occupancy_grid):
    return occupancy_grid.get_occupancy(self._pose[:2])


  def move(self, dt):
    delta_pose = np.zeros(3, dtype=np.float32)

    u =  np.random.random_sample()*(U_MAX*2) - U_MAX
    w =  np.random.random_sample()*(W_MAX*2) - W_MAX
    #print("u, w",u, w)

    delta_pose[X] += u * dt
    delta_pose[Y] += 0
    delta_pose[YAW] += w * dt

    forward_vel = np.abs(delta_pose[X])+0.01
    rot_vel = np.abs(delta_pose[YAW])+0.01
    #print("f, r",forward_vel, rot_vel)
    world_delta_pose = delta_pose.copy()

    # Apply motion model
    if forward_vel > 0:
      #print("forward", forward_vel, self._weight, forward_vel * self._weight)
      world_delta_pose[X] = np.random.normal(world_delta_pose[X], forward_vel * self._weight)
      world_delta_pose[Y] = np.random.normal(world_delta_pose[Y], forward_vel * self._weight)
    if rot_vel > 0:
      #print(rot_vel, self._weight, rot_vel * self._weight)
      world_delta_pose[YAW] = np.random.normal(world_delta_pose[YAW], rot_vel * self._weight)

    # Transform to world coords
    predicted_yaw = self._pose[YAW] + world_delta_pose[YAW] / 40
    s, c = np.sin(predicted_yaw), np.cos(predicted_yaw)
    rot_mat = np.array(((c, -s), (s, c)))
    world_delta_pose[0:2] = np.matmul(rot_mat, delta_pose[0:2])

    self._pose += world_delta_pose
    

  def compute_weight(self, measured_pose, variance, occupancy_grid, policelocations):
    if not self.is_valid(occupancy_grid):
	self._weight = 0
    elif not in_line_of_sight(self._pose, policelocations, occupancy_grid):
        self._weight = 0
    elif(variance == float('inf')):
        self._weight = 1
    else:
	weights = np.zeros(3, dtype=np.float32)

    	sigma = np.sqrt(variance)
    	robot_measurements = measured_pose
    	particle_measurements = self._pose

        
        weights = norm.pdf(particle_measurements, robot_measurements, sigma)
        self._weight = np.prod(weights+1) -1
	

        print("mp: ", measured_pose, " var: ", variance," self_pose: ", self._pose) 
        print("weight updated to: ", self._weight)
        


particle_publisher = None
frame_id = None
def initialize():
  global particle_publisher
  global frame_id
  particle_publisher = rospy.Publisher('/particles', PointCloud, queue_size=1)
  frame_id = 0

def update_particles(particles, dt, measured_pose, variance, num_particles, occupancy_grid):
  global particle_publisher
  global frame_id

  # Update particle positions and weights.
  total_weight = 0.
  for i, p in enumerate(particles):
    p.move(dt)
    p.compute_weight(measured_pose, variance, occupancy_grid)
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
  frame_id += 1
  
  return new_particles

'''
def run(args):
  rospy.init_node('localization')

  # Update control every 100 ms.
  rate_limiter = rospy.Rate(100)
  publisher = rospy.Publisher('/cmd_vel', Twist, queue_size=5)
  particle_publisher = rospy.Publisher('/particles', PointCloud, queue_size=1)
  laser = SimpleLaser()
  motion = Motion()
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

    u, w = braitenberg(*laser.measurements)
    vel_msg = Twist()
    vel_msg.linear.x = u
    vel_msg.angular.z = w
    publisher.publish(vel_msg)

    # Update particle positions and weights.
    total_weight = 0.
    delta_pose = motion.delta_pose
    for i, p in enumerate(particles):
      p.move(delta_pose)
      p.compute_weight(*laser.measurements)
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
'''
