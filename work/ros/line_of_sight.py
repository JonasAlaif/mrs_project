import numpy as np

# Has been copied to obstacle_avoidance.py, this method is no longer used
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