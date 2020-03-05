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

  clients = dict()
  client_path_tuples = dict()

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

        # used for rrt
        time_now = float(rospy.Time.now().to_sec())
        client_path_tuples[s] = (None,
                                 rrt.sample_random_position(occupancy_grid_base),
                                 time_now)
        print("registered socket", s, "with name", name)

    for s in xlist:
      print("removing socket ", s, " with name ", clients[s][0])
      del clients[s]

    # create an occupancy grid containing the original grid and the robots (to avoid collisions)
    # this is currently unused
    occupancy_grid = rrt.OccupancyGrid(occupancy_grid_base.values,
                                       np.append(occupancy_grid_base.origin, 0),
                                       occupancy_grid_base.resolution)
    for sock in clients.keys():
      centre = clients[sock][4].pose[:2]
      centre_indexes = occupancy_grid.get_index(centre)
      for i in range(-1, 1):
        for j in range(-1, 1):
          occupancy_grid.values[centre_indexes[0] + i][centre_indexes[1] + j] = rrt.OCCUPIED

    # split the bots into goodies and baddies
    police = []
    baddies = []
    for client in clients.keys():
      if clients[client][1] == 'baddie':
        baddies.append(client)
      elif clients[client][1] == 'police':
        police.append(client)

    # check if any police have caught any baddies
    caught_baddies = []
    for polbot in  police:
      for badbot in baddies:
        dist_from_baddie = np.linalg.norm(clients[polbot][4].pose[:2] - clients[badbot][4].pose[:2])
        if dist_from_baddie < 0.3:
          print(clients[polbot][0], "caught baddie", clients[badbot][0], "!!!!!!!!!!!!!!!!")

          # stop this baddie
          vel_msg = Twist()
          vel_msg.linear.x = 0
          vel_msg.angular.z = 0
          clients[badbot][2].publish(vel_msg)

          # mark this baddie as caught
          caught_baddies.append(badbot)

    # remove baddies
    # remove this baddie from the list of clients
    for badbot in caught_baddies:
      name = clients[badbot][0]
      del clients[badbot]
      del client_path_tuples[badbot]

      # for now, also delete the baddie model
      rospy.ServiceProxy('gazebo/delete_model', DeleteModel)('turtlebot3_burger_' + name)

    # pick a baddie for all the police to chase
    baddie = None
    for sock in clients.keys():
      if clients[sock][1] == "baddie":
        baddie = sock


    # now update the currently connected robots
    for sock in clients.keys():
      (name, role, pub, laser, gtpose) = clients[sock]
      #if not laser.ready:
      #  continue
      (path, goal, time_created) = client_path_tuples[sock]

      # check if the goal has been reached
      goal_reached = np.linalg.norm(gtpose.pose[:2] - goal) < .2
      time_now = rospy.Time.now().to_sec()

      baddie_dist_from_goal = None
      if baddie is not None and role == 'police':
        baddie_dist_from_goal = np.linalg.norm(goal - clients[baddie][4].pose[:2])
        print("baddie dist:", baddie_dist_from_goal)

      # if the robot has reached the goal or there was no path
      if goal_reached or path is None or len(path) == 0 or (role == 'police' and baddie_dist_from_goal > 0.30):
        if baddie_dist_from_goal > 0.3:
          print("baddie moved")
        if gtpose.ready:
          new_goal = None
          if baddie is not None and role == 'police':
            # choose the baddie's location as the goal
            new_goal = np.array(clients[baddie][4].pose[:2])
          else:
            # choose a new random goal and get a new path
            new_goal = rrt.sample_random_position(occupancy_grid)

          print("goal:", new_goal)
          start_node, end_node = rrt.rrt(gtpose.pose, new_goal, occupancy_grid_base, MAX_ITERATIONS)
          #print("start node:", start_node, "end node:", end_node)
          new_path = rrt_navigation.get_path(end_node)
          path = new_path
          client_path_tuples[sock] = (new_path, new_goal, time_now)
          print(name, "reached goal, giving it the new goal:", new_goal)
        else:
          print("ground truth not ready")
          continue
      # update the path every 10 seconds
      elif time_now - time_created > 10:
        start_node, end_node = rrt.rrt(gtpose.pose, goal, occupancy_grid, MAX_ITERATIONS)
        new_path = rrt_navigation.get_path(end_node)
        path = new_path
        client_path_tuples[sock] = (new_path, goal, time_now)
        #print("updating the path for", name)



      #print("gtpose: ", gtpose.pose)
      lin_pos = np.array([gtpose.pose[X] + EPSILON*np.cos(gtpose.pose[YAW]),\
                          gtpose.pose[Y] + EPSILON*np.sin(gtpose.pose[YAW])])
      #print("path: ", path)
      #v = rrt_navigation.get_velocity(lin_pos, np.array(path, dtype=np.float32))
      if role == 'police':
        v = rrt_navigation.get_velocity(lin_pos, np.array(path, dtype=np.float32), speed=SPEED*2)
      else:
        v = rrt_navigation.get_velocity(lin_pos, np.array(path, dtype=np.float32), speed=SPEED)
      #print("vel: ", v)
      #u, w = rrt_navigation.feedback_linearized(gtpose.pose, v, epsilon=EPSILON, SPEED)
      if role == 'police':
        u, w = rrt_navigation.feedback_linearized(gtpose.pose, v, epsilon=EPSILON, speed=SPEED*2)
      else:
        u, w = rrt_navigation.feedback_linearized(gtpose.pose, v, epsilon=EPSILON, speed=SPEED)




      #u, w = avoidance_method(*laser.measurements)
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
