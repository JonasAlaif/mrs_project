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

# Robot motion commands:
# http://docs.ros.org/api/geometry_msgs/html/msg/Twist.html
from geometry_msgs.msg import Twist

directory = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, directory)
try:
  import obstacle_avoidance
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

def run(args):
  avoidance_method = getattr(obstacle_avoidance, args.mode)
  rospy.init_node('controller')
  occupancy_grid = None
  if args.map is not None:
    # Load map.
    with open(args.map + '.yaml') as fp:
      filedata = yaml.load(fp)
    img = rrt.read_pgm(os.path.join(os.path.dirname(args.map), filedata['image']))
    occupancy_grid = np.empty_like(img, dtype=np.int8)
    occupancy_grid[:] = UNKNOWN
    occupancy_grid[img < .1] = OCCUPIED
    occupancy_grid[img > .9] = FREE
    # Transpose (undo ROS processing).
    occupancy_grid = occupancy_grid.T
    # Invert Y-axis.
    occupancy_grid = occupancy_grid[:, ::-1]
    occupancy_grid = rrt.OccupancyGrid(occupancy_grid, filedata['origin'], filedata['resolution'])



  servsocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  servsocket.bind(('', int(args.port)))
  print("listening on port ", int(args.port))
  servsocket.setblocking(False)
  servsocket.listen(5)

  clients = {}

  # Update controller 20 times a second
  rate_limiter = rospy.Rate(20)
  while not rospy.is_shutdown():
    # first look for new incoming connections
    try:
      newsock, newaddr = servsocket.accept()
      clients[newsock] = (None, None, None, None, None)
    except socket.error:
      # we didn't get a new connection
      pass

    # check if any sockets are readable or have closed
    rlist, wlist, xlist = select.select(clients.keys(), [], clients.keys(), 0)
    for s in rlist:
      data = s.recv(1024)
      if not len(data) == 0:
        split = data.split(' ')
        name = split[0]
        role = split[1]
        clients[s] = (name, \
                      role, \
                      rospy.Publisher('/' + name + '/cmd_vel', Twist, queue_size=5), \
                      obstacle_avoidance.SimpleLaser(name), \
                      obstacle_avoidance.GroundtruthPose('turtlebot3_burger_' + name) \
                     )
        print("registered socket", s, "with name", name)

    for s in xlist:
      print("removing socket ", s, " with name ", clients[s][0])
      del clients[s]

    # now update the currently connected robots
    for client in clients.values():
      (name, role, pub, laser, gtpose) = client
      if not laser.ready:
        continue

      u, w = avoidance_method(*laser.measurements)
      vel_msg = Twist()
      vel_msg.linear.x = u
      vel_msg.angular.z = w
      pub.publish(vel_msg)

    # sleep so we don't overutilise the CPU
    rate_limiter.sleep()

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Controls the robots')
  parser.add_argument('--mode', action='store', default='braitenberg', help='Method.', choices=['braitenberg', 'rule_based'])
  parser.add_argument('--port', action='store', default='12321', help='Port.')
  parser.add_argument('--map', action='store', default=None, help='Map.')
  args, unknown = parser.parse_known_args()
  run(args)
