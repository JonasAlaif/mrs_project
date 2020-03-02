#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import rospy
import select
import socket
import sys
import time

# Robot motion commands:
# http://docs.ros.org/api/geometry_msgs/html/msg/Twist.html
from geometry_msgs.msg import Twist

directory = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, directory)
try:
  import obstacle_avoidance
except ImportError:
  raise ImportError('Unable to import obstacle_avoidance.py. Make sure this file is in "{}"'.format(directory))

def run(args):
  avoidance_method = getattr(obstacle_avoidance, args.mode)
  rospy.init_node('controller')

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
      clients[newsock] = (None, None, None)
    except socket.error:
      # we didn't get a new connection
      pass

    # check if any sockets are readable or have closed
    rlist, wlist, xlist = select.select(clients.keys(), [], clients.keys(), 0)
    for s in rlist:
      data = s.recv(1024)
      if not len(data) == 0:
        clients[s] = (data, rospy.Publisher('/' + data + '/cmd_vel', Twist, queue_size=5), obstacle_avoidance.SimpleLaser(data))
        print("registered socket", s, "with name", data)

    for s in xlist:
      print("removing socket ", s, " with name ", clients[s][0])
      del clients[s]

    # now update the currently connected robots
    for client in clients.values():
      (name, pub, laser) = client
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
  args, unknown = parser.parse_known_args()
  run(args)
