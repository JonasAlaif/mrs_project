#!/usr/bin/env python

import argparse
import socket
import time

def register(name, role):
  s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  connected = False
  while not connected:
    try:
      s.connect(('localhost', 12321))
      connected = True
    except socket.error:
      print("couldn't register", name)
      time.sleep(0.5)
      pass
  s.send(name + " " + role)
  print("registered name ", name)
  s.close()


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Runs obstacle avoidance')
  parser.add_argument('--name', action='store', default='turtlebot3_0', help='Name.')
  parser.add_argument('--role', action='store', default='police', help='Role.', choices=['police', 'baddie'])
  parser.add_argument('--all', action='store', default='false', help='Register all.', choices=['true', 'false'])
  args, unknown = parser.parse_known_args()

  if args.all == 'true':
    register('turtlebot_1', 'police')
    register('turtlebot_2', 'police')
    register('turtlebot_5', 'police')
    register('turtlebot_6', 'police')
    register('turtlebot_3', 'baddie')
    register('turtlebot_4', 'baddie')
  else:
    register(args.name, args.role)
