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

import gazebo_msgs.srv
from std_srvs.srv import Empty, EmptyRequest

# Robot motion commands:
# http://docs.ros.org/api/geometry_msgs/html/msg/Twist.html
from geometry_msgs.msg import Twist

directory = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, directory)
try:
  import obstacle_avoidance
  import rrt_navigation
except ImportError:
  raise ImportError('Unable to import obstacle_avoidance.py. Make sure this file is in "{}"'.format(directory))

directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../python')
sys.path.insert(0, directory)
try:
  import rrt
  import rrt_improved
  import potential_field_map
except ImportError:
  raise ImportError('Unable to import obstacle_avoidance.py. Make sure this file is in "{}"'.format(directory))

MAX_ITERATIONS = 1500
EPSILON = 0.15

X = 0
Y = 1
YAW = 2

SPEED = 0.2

#EXIT_POSITION = np.array([9.0, 0.0])
EXIT_POSITION = np.array([9.0, 0.0, 0.0])

obstacle_map = None
pause_srv = None
unpause_srv = None

def initialize():
  global obstacle_map
  global pause_srv
  global unpause_srv
  local_dir = os.path.dirname(os.path.abspath(__file__))
  obstacle_map = potential_field_map.initialize(local_dir + '/../python/map_city_3')
  pause_srv = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
  print('pause_srv:', pause_srv)
  unpause_srv = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
  print('unpause_srv:', unpause_srv)
  #potential_field_map.display_obst_map(obstacle_map)

'''
TODO upgrade this documentation if the description of this function changes
navigate_baddie takes a series of arguments and gives you back the straight-line and angular velocities
that you should pass to the robot in order to follow the path.
This function will select a random goal for the baddie each time it reaches its current goal.

Arguments:
    name - the name of the robot being navigated
    gtpose - the GroundtruthPose of the robot being navigated
    laser - the SimpleLaser of the robot being navigated
    paths - the dictionary of (path, goal, time_created) tuples of all objects
            only the value of paths[name] is used and updated in this function
    occupancy_grid - the occupancy grid of the map
    max_iterations - the maximum number of times the rrt function should iterate
'''
def navigate_baddie_rrt(name, laser, gtpose, paths, occupancy_grid, max_iterations):
  (path, goal, time_created) = paths[name]

  # get curret time (for updating path)
  time_now = rospy.Time.now().to_sec()
  # check if current goal has been reached
  if gtpose.ready:
    goal_reached = np.linalg.norm(gtpose.pose[:2] - goal[:2]) < 0.3
  else:
    goal_reached = None

  # generate a new goal if needed
  new_goal = None
  if goal_reached or path is None or len(path) == 0:
    # generate a new random goal
    new_goal = rrt.sample_random_position(occupancy_grid)
    goal = new_goal

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

    v = rrt_navigation.get_velocity(lin_pos, np.array(path, dtype=np.float32), speed=SPEED)
    u, w = rrt_navigation.feedback_linearized(gtpose.pose, v, epsilon=EPSILON, speed=SPEED)
    return u, w
  else:
    return None, None


def navigate_baddie_hybrid(name, laser, gtpose, paths, occupancy_grid, max_iterations, police):
  (path, goal, time_created) = paths[name]
  #print('navigating baddie')

  # get curret time (for updating path)
  time_now = rospy.Time.now().to_sec()
  # check if current goal has been reached
  if gtpose.ready:
    goal_reached = np.linalg.norm(gtpose.pose[:2] - goal[:2]) < 0.3
  else:
    goal_reached = None

  # generate a new goal if needed
  new_goal = None
  if goal_reached or path is None or len(path) == 0:
    # generate a new random goal
    #new_goal = rrt_improved.sample_random_position(occupancy_grid)[:2]
    new_goal = EXIT_POSITION
    goal = new_goal

  # if we selected a new goal or it's been a while since we last calculated a path, update our path
  if new_goal is not None or (time_now - time_created > 3 and goal is not None):
    if gtpose.ready:
      recalc = True
      if new_goal is None:
        # we're updating due to time having passed
        # just check if our path has police nearby
        # if not, don't bother changing it
        weight = 1
        min_dist_all = float('inf')
        for i in range(len(path) - 1):
          start = path[i]
          end = path[i+1]
          # for this stretch, check if there is a police bot near it
          min_dist = float('inf')
          for p in police:
            dist = rrt_improved.seg_dist(start[:2], end[:2], p[0][:2])
            if dist < min_dist:
              min_dist = dist

          min_dist = min_dist/weight
          if min_dist < min_dist_all:
            min_dist_all = min_dist
          weight = weight/np.linalg.norm(end - start)
        if min_dist_all < 1.5:
          print(name, 'recalculating path due to nearby police')
          print(min_dist_all)
          recalc = True
        else:
          print(name, 'no recalculation required')
          print(min_dist_all)
          paths[name] = (path, goal, time_now)
          recalc = False

      if recalc:
        # recalculate the path
        global pause_srv
        pause_srv(EmptyRequest())
        print('paused')

        start_node, end_node = rrt_improved.rrt_nocircle(gtpose.pose, goal, occupancy_grid, police, max_iterations)

        global unpause_srv
        unpause_srv(EmptyRequest())
        print('unpaused')

        print(end_node is None)
        new_path = rrt_navigation.get_path(end_node)
        paths[name] = (new_path, goal, time_now)

        if new_path is not None:
          if end_node is not None:
            #print(new_path)
            print('new path cost:', end_node.cost)
          path = new_path
          print('path updated for', name)

          #pause_srv(EmptyRequest())
          # Plot environment.
          #fig, ax = plt.subplots()
          #occupancy_grid.draw()
          #rrt_improved.draw_solution(start_node, police, end_node)
          #plt.scatter(gtpose.pose[0], gtpose.pose[1], s=10, marker='o', color='blue', zorder=1000)
          #plt.scatter(goal[0], goal[1], s=10, marker='o', color='red', zorder=1000)

          #plt.axis('equal')
          #plt.xlabel('x')
          #plt.ylabel('y')
          #plt.xlim([-.5 - 2., 2. + .5])
          #plt.ylim([-.5 - 2., 2. + .5])
          #plt.show()
          #unpause_srv(EmptyRequest())
    else:
      print(name, 'ground truth not ready for goal setting')

  if path is not None and len(path) > 0 and gtpose.ready:
    # find distance to first element in path
    lin_pos = np.array([gtpose.pose[X] + EPSILON*np.cos(gtpose.pose[YAW]),
                        gtpose.pose[Y] + EPSILON*np.sin(gtpose.pose[YAW]),
                        gtpose.pose[YAW]])
    dist = np.linalg.norm(lin_pos[:2] - path[0][:2])
    #print(dist)
    if dist < 0.3:
      print('got close')
      print('gtpose', gtpose.pose[:2], 'next', path[0])
      # delete the current point from the path
      del path[0]
      #print(path)
      # update the paths
      paths[name] = (path, goal, time_now)
      if len(path) == 0:
        print('got to the end of the path')
        return None, None

    v = potential_field_map.get_velocity(lin_pos[:2], [(path[0][:2], 1)], [], obstacle_map)
    #print(name, 'potential field velocity', v)
    u, w = rrt_navigation.feedback_linearized(gtpose.pose, v, epsilon=EPSILON, speed=SPEED)
    return u, w
  elif gtpose.ready:
    #class Struct(object): pass
    #goalstruct = Struct()
    #goalstruct.pose = goal[:2]
    return navigate_baddie_pot_nai(name, laser, gtpose, goal[:2], paths, occupancy_grid, max_iterations, police)
  else:
    return None, None


def sigmoid(x):
  return np.tanh(x/2)

def baddie_braitenberg(name, laser, gtpose, paths, occupancy_grid, max_iterations, police):
  (front, front_left, front_right, left, right) = laser.measurements
  # u in [m/s]
  # w in [rad/s] going counter-clockwise.
  closestDistSide = 0.05
  closestDistFront = 0.4
  u = sigmoid(front-closestDistFront) * sigmoid(front_left+closestDistSide) * sigmoid(front_right+closestDistSide) * 1
  w = sigmoid(5/front_right + 1/sigmoid(right) - 5/front_left - 1/sigmoid(left))
  return u, w

  
def navigate_baddie_pot_nai(name, laser, gtpose, goal, paths, occupancy_grid, max_iterations, other_police):
  if goal is None:
    return 0, 0
  global obstacle_map
  control_pos = gtpose.pose[:2] + np.array([EPSILON*np.cos(gtpose.pose[YAW]), EPSILON*np.sin(gtpose.pose[YAW])]) / 3
  v = potential_field_map.get_velocity(control_pos, goal, other_police, obstacle_map, mode='all')
  u, w = rrt_navigation.feedback_linearized(gtpose.pose, v, epsilon=EPSILON, speed=SPEED)
  #print('My pos: ', control_pos)
  #print('Target pos: ', baddie_gtp.pose[:2])
  #print('Direction pos: ', v)
  return u, w



















