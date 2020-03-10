#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
import random
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
  import baddie_navigation
  import police_navigation
except ImportError:
  raise ImportError('Unable to import obstacle_avoidance.py. Make sure this file is in "{}"'.format(directory))

directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../python')
sys.path.insert(0, directory)
try:
  import rrt
except ImportError:
  raise ImportError('Unable to import obstacle_avoidance.py. Make sure this file is in "{}"'.format(directory))

# Constants for occupancy grid.
FREE = 0
UNKNOWN = 1
OCCUPIED = 2

MAX_ITERATIONS = 1500
EPSILON = 0.1

X = 0
Y = 1
YAW = 2

SPEED = 0.1

def run(args):
  obstacle_map = police_navigation.initialize()
  avoidance_method = getattr(obstacle_avoidance, args.mode)
  rospy.init_node('controller')
  occupancy_grid_base = None
  if args.map is not None:
    # Load map.
    with open(args.map + '.yaml') as fp:
      filedata = yaml.load(fp)
    img = rrt.read_pgm(os.path.join(os.path.dirname(args.map), filedata['image']))
    occupancy_grid_base = np.empty_like(img, dtype=np.int8)
    occupancy_grid_base[:] = UNKNOWN
    occupancy_grid_base[img < .1] = OCCUPIED
    occupancy_grid_base[img > .9] = FREE
    # Transpose (undo ROS processing).
    occupancy_grid_base = occupancy_grid_base.T
    # Invert Y-axis.
    occupancy_grid_base = occupancy_grid_base[:, ::-1]
    occupancy_grid_base = rrt.OccupancyGrid(occupancy_grid_base, filedata['origin'], filedata['resolution'])


  servsocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  servsocket.bind(('', int(args.port)))
  print("listening on port ", int(args.port))
  servsocket.setblocking(False)
  servsocket.listen(5)

  # list of accepted sockets
  sockets = list()


  # these will be dictionaries indexed on the robot name which give you back the 4-tuple
  # (Publisher, SimpeLaser, GroundtruthPose)
  police = dict()
  baddies = dict()

  # this will be a dictionary indexed on the robot name which gives you back the 3-tuple
  # (path, goal, time_created)
  # this is used for RRT
  client_path_tuples = dict()

  # the target that the police will chase
  target = None

  # Update controller 10 times a second
  rate_limiter = rospy.Rate(10)
  while not rospy.is_shutdown():
    # first look for new incoming connections
    try:
      newsock, newaddr = servsocket.accept()
      sockets.append(newsock)
      print('new socket created')
    except socket.error:
      # we didn't get a new connection
      pass

    # check if any sockets are readable
    rlist, wlist, xlist = select.select(sockets, [], sockets, 0)
    for s in rlist:
      data = s.recv(1024)
      if not len(data) == 0:
        split = data.split(' ')
        name = split[0]
        role = split[1]
        temp = (rospy.Publisher('/' + name + '/cmd_vel', Twist, queue_size=5),
                         obstacle_avoidance.SimpleLaser(name),
                         obstacle_avoidance.GroundtruthPose('turtlebot3_burger_' + name)
                        )
        if role == 'police':
          police[name] = temp
        else:
          baddies[name] = temp

        # used for rrt
        time_now = float(rospy.Time.now().to_sec())
        client_path_tuples[name] = (None,
                                    np.array([9.0, 0.0]),
                                    time_now)
        print("registered robot", name, "with role", role)

    # create an occupancy grid containing the original grid and the robots (to avoid collisions)
    # this is currently unused
    occupancy_grid = rrt.OccupancyGrid(occupancy_grid_base.values,
                                       np.append(occupancy_grid_base.origin, 0),
                                       occupancy_grid_base.resolution)
    for name in police.keys():
      # get the location of the bot
      centre = police[name][2].pose[:2]
      centre_indexes = occupancy_grid.get_index(centre)
      for i in range(-1, 1):
        for j in range(-1, 1):
          occupancy_grid.values[centre_indexes[0] + i][centre_indexes[1] + j] = rrt.OCCUPIED
    for name in baddies.keys():
      # get the location of the bot
      centre = baddies[name][2].pose[:2]
      centre_indexes = occupancy_grid.get_index(centre)
      for i in range(-1, 1):
        for j in range(-1, 1):
          occupancy_grid.values[centre_indexes[0] + i][centre_indexes[1] + j] = rrt.OCCUPIED

    # check if any police have caught any baddies
    caught_baddies = []
    for polname in police.keys():
      for badname in baddies.keys():
        dist_from_baddie = np.linalg.norm(police[polname][2].pose[:2] - baddies[badname][2].pose[:2])
        if dist_from_baddie < 0.3:
          print(polname, "caught baddie", badname, "!!!!!!!!!!!!!!!!")

          # stop this baddie
          vel_msg = Twist()
          vel_msg.linear.x = 0
          vel_msg.angular.z = 0
          baddies[badname][0].publish(vel_msg)

          # mark this baddie as caught
          caught_baddies.append(badname)

    # remove baddies that were caught
    for badname in caught_baddies:
      del baddies[badname]
      del client_path_tuples[badname]
      if badname == target:
        target = None

      # for now, also delete the baddie model
      rospy.ServiceProxy('gazebo/delete_model', DeleteModel)('turtlebot3_burger_' + badname)

    # pick a baddie for all the police to chase if there isn't already one
    if len(baddies.keys()) > 0 and target == None:
      for name in random.sample(baddies.keys(), len(baddies.keys())):
        # TODO change baddie selection policy (closest/furthest/something else?)
        if baddies[name][2].ready:
          target = name
          break
      else:
        target = None

    baddie_gtpose = baddies[target][2] if target is not None else None
    baddie_pose = baddie_gtpose.observed_pose([pol[2].pose for pol in police.values()], obstacle_map) if target is not None else None
    # update police navigation
    for name in police.keys():
      (pub, laser, gtpose) = police[name]
      (path, goal, time_created) = client_path_tuples[name]
      other_police = dict(police)
      del other_police[name]
      other_police_pos = [(pol[2].pose[:2], 0.5) for pol in other_police.values()]
      u, w = 0, 0
      if baddie_pose != None and baddie_pose[1] > 0.1:
        u, w = police_navigation.navigate_police_2(name,
                                               gtpose,
                                               laser,
                                               baddie_pose[0],
                                               client_path_tuples,
                                               occupancy_grid_base,
                                               MAX_ITERATIONS, other_police_pos)
      if u is not None and w is not None:
        vel_msg = Twist()
        vel_msg.linear.x = u
        vel_msg.angular.z = w
        pub.publish(vel_msg)

    # update baddie navigation
    for name in baddies.keys():
      (pub, laser, gtpose) = baddies[name]
      (path, goal, time_created) = client_path_tuples[name]
      '''
      u, w = baddie_navigation.navigate_baddie(name,
                                               laser,
                                               gtpose,
                                               client_path_tuples,
                                               occupancy_grid,
                                               MAX_ITERATIONS)'''
      police_pos = [(pol[2].pose[:2], 1.6) for pol in police.values()]
      u, w = police_navigation.navigate_police_2(name,
                                               gtpose,
                                               laser,
                                               goal,
                                               client_path_tuples,
                                               occupancy_grid,
                                               MAX_ITERATIONS, police_pos)

      if u is not None and w is not None:
        vel_msg = Twist()
        vel_msg.linear.x = 1.2 * u
        vel_msg.angular.z = w
        pub.publish(vel_msg)


    # sleep so we don't overutilise the CPU
    #rate_limiter.sleep()

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Controls the robots')
  parser.add_argument('--mode', action='store', default='braitenberg', help='Method.', choices=['braitenberg', 'rule_based'])
  parser.add_argument('--port', action='store', default='12321', help='Port.')
  parser.add_argument('--map', action='store', default=None, help='Map.')
  args, unknown = parser.parse_known_args()
  run(args)
