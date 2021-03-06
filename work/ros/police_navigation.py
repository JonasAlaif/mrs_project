#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
import rospy
import select
import socket
import sys
import time
import yaml

import matplotlib.pylab as plt

# Robot motion commands:
# http://docs.ros.org/api/geometry_msgs/html/msg/Twist.html
from geometry_msgs.msg import Twist
from gazebo_msgs.srv import DeleteModel

directory = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, directory)
try:
  import obstacle_avoidance
  import rrt_navigation
except ImportError:
  raise ImportError('Unable to import obstacle_avoidance.py. Make sure this file is in "{}"'.format(directory))

directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../python')
sys.path.insert(0, directory)
import rrt
import potential_field_map

MAX_ITERATIONS = 1500
EPSILON = 0.15

X = 0
Y = 1
YAW = 2

SPEED = 0.1

obstacle_map = None

def initialize():
  global obstacle_map
  local_dir = os.path.dirname(os.path.abspath(__file__))
  obstacle_map = potential_field_map.initialize(local_dir + '/../python/map_city_3')
  #potential_field_map.display_obst_map(obstacle_map)
  return obstacle_map


'''
TODO upgrade this documentation if the description of this function changes
navigate_police takes a series of arguments and gives you back the straight-line and angular velocities
that you should pass to the robot in order to follow the path.
This function sets the given baddie_gtp as the goal position if it is not None.
If it is None, then it picks a random location

Arguments:
    name - the name of the robot being navigated
    gtpose - the GroundtruthPose of the robot being navigated
    laser - the SimpleLaser of the robot being navigated
    baddie_gtp - the GroundtruthPose of the baddie being tracked
    paths - the dictionary of (path, goal, time_created) tuples of all objects
            only the value of paths[name] is used and updated in this function
    occupancy_grid - the occupancy grid of the map
    max_iterations - the maximum number of times the rrt function should iterate
'''
def navigate_police(name, gtpose, laser, baddie_gtp, paths, occupancy_grid, max_iterations, other_police, max_speed):
  (path, goal, time_created) = paths[name]

  # get curret time (for updating path)
  time_now = rospy.Time.now().to_sec()
  # check if current goal has been reached
  if gtpose.ready:
    goal_reached = np.linalg.norm(gtpose.pose[:2] - goal) < 0.2
  else:
    goal_reached = None

  baddie_dist_from_goal = None
  if baddie_gtp is not None and baddie_gtp.ready:
    baddie_dist_from_goal = np.linalg.norm(baddie_gtp.pose[:2] - goal)

  # generate a new goal if needed
  new_goal = None
  if goal_reached or path is None or len(path) == 0:
    # generate a new goal
    if baddie_gtp is not None and baddie_gtp.ready:
      # put the goal where the baddie is
      new_goal = list(baddie_gtp.pose[:2])
      goal = new_goal
      print('new target goal:', new_goal)
    else:
      # generate a random goal
      new_goal = rrt.sample_random_position(occupancy_grid)
      goal = new_goal
      print('new random goal:', new_goal)

  elif baddie_dist_from_goal > 0.3:
    # the baddie moved away from the goal. generate a new goal to follow them
    print('baddie moved')
    print('dist: ', baddie_dist_from_goal)
    new_goal = list(baddie_gtp.pose[:2])
    goal = new_goal
    print('new target goal:', new_goal)

  # if we selected a new goal or it's been a while since we last calculated a path, update our path
  if new_goal is not None or (time_now - time_created > 10 and goal is not None):
    if gtpose.ready:
      start_node, end_node = rrt.rrt(gtpose.pose, goal, occupancy_grid, max_iterations)
      new_path = rrt_navigation.get_path(end_node)
      paths[name] = (new_path, goal, time_now)
      if new_path is not None:
        path = new_path
        print('path updated for', name)
    else:
      print(name, 'ground truth not ready for goal setting')

  if path is not None and len(path) > 0 and gtpose.ready:
    lin_pos = np.array([gtpose.pose[X] + EPSILON*np.cos(gtpose.pose[YAW]),\
                        gtpose.pose[Y] + EPSILON*np.sin(gtpose.pose[YAW])])

    v = rrt_navigation.get_velocity(lin_pos, np.array(path, dtype=np.float32), speed=max_speed)
    u, w = rrt_navigation.feedback_linearized(gtpose.pose, v, epsilon=EPSILON, speed=max_speed)
    return u, w
  else:
    return None, None


def navigate_police_2(name, laser, gtpose, baddie_poses, paths, occupancy_grid, max_iterations, other_police, max_speed):
  global obstacle_map
  control_pos = gtpose.pose[:2] + np.array([EPSILON*np.cos(gtpose.pose[YAW]), EPSILON*np.sin(gtpose.pose[YAW])]) / 3
  v = potential_field_map.get_velocity(control_pos, baddie_poses, other_police, obstacle_map)
  u, w = rrt_navigation.feedback_linearized(gtpose.pose, v, epsilon=EPSILON, speed=max_speed)
  #print('My pos: ', control_pos)
  #print('Target pos: ', baddie_gtp.pose[:2])
  #print('Direction pos: ', v)
  return u, w
