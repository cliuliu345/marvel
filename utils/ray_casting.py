"""CPU ray kernels. Keep traversal order and endpoint rules; no fast math."""
import numpy as np
from numba import njit


@njit(cache=True, boundscheck=True)
def collision_check(x0, y0, x1, y1, ground_truth, robot_belief):
    x0 = int(np.round(x0))
    y0 = int(np.round(y0))
    x1 = int(np.round(x1))
    y1 = int(np.round(y1))
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    x, y = x0, y0
    error = dx - dy
    x_inc = 1 if x1 > x0 else -1
    y_inc = 1 if y1 > y0 else -1
    dx *= 2
    dy *= 2
    collision_flag = 0
    max_collision = 2
    while 0 <= x < ground_truth.shape[1] and 0 <= y < ground_truth.shape[0]:
        k = ground_truth[y, x]
        if k == 1 and collision_flag < max_collision:
            collision_flag += 1
            if collision_flag >= max_collision:
                break
        if k != 1 and collision_flag > 0:
            break
        if x == x1 and y == y1:
            break
        robot_belief[y, x] = k
        if error > 0:
            x += x_inc
            error -= dy
        else:
            y += y_inc
            error += dx
    return robot_belief


@njit(cache=True, boundscheck=True)
def sensor_sweep(x0, y0, endpoints_x, endpoints_y, ground_truth, robot_belief):
    for i in range(len(endpoints_x)):
        collision_check(x0, y0, endpoints_x[i], endpoints_y[i], ground_truth, robot_belief)
    return robot_belief


@njit(cache=True, boundscheck=True)
def check_collision_cells(start_cell, end_cell, map_array, occupied, unknown):
    x0, y0 = start_cell[0], start_cell[1]
    x1, y1 = end_cell[0], end_cell[1]
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    x, y = x0, y0
    error = dx - dy
    x_inc = 1 if x1 > x0 else -1
    y_inc = 1 if y1 > y0 else -1
    dx *= 2
    dy *= 2
    while 0 <= x < map_array.shape[1] and 0 <= y < map_array.shape[0]:
        k = map_array[int(y), int(x)]
        # The original excludes the destination cell from collision checks.
        if x == x1 and y == y1:
            break
        if k == occupied or k == unknown:
            return True
        if error > 0:
            x += x_inc
            error -= dy
        else:
            y += y_inc
            error += dx
    return False
