#!/usr/bin/env python

import argparse
import socket
import time

def run(args):
  name = args.name
  role = args.role
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
  args, unknown = parser.parse_known_args()
  run(args)
