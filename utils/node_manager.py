"""
Manages nodes in a quadtree-based spatial graph for exploration and frontier tracking.

This class handles the creation, updating, and management of nodes representing 
spatial locations during exploration. It supports tracking observable frontiers, 
calculating node utilities, managing neighbor relationships, and performing 
pathfinding algorithms like Dijkstra and A* on the node graph.

Key functionalities:
- Add, remove, and update nodes in a quadtree data structure
- Calculate node utilities based on observable frontiers
- Manage neighbor node connections
- Perform graph-based pathfinding algorithms
"""
import time

import numpy as np
from utils.utils import *
from parameter import *
import utils.quads as quads
from utils.graph_search import dijkstra_distances


class NodeManager:
    def __init__(self, fov, sensor_range, utility_range=None, plot=False):
        self.nodes_dict = quads.QuadTree((0, 0), 1000, 1000)
        self.plot = plot
        self.fov = fov
        self.sensor_range = sensor_range
        self.frontier_version = 0
        self.global_frontiers = frozenset()
        if utility_range is None:
            self.utility_range = UTILITY_RANGE
        else:
            self.utility_range = utility_range

    def check_node_exist_in_dict(self, coords):
        key = (coords[0], coords[1])
        exist = self.nodes_dict.find(key)
        return exist

    def add_node_to_dict(self, coords, local_frontiers, updating_map_info):
        key = (coords[0], coords[1])
        node = Node(coords, local_frontiers, updating_map_info, self.fov, self.sensor_range, utility_range=self.utility_range)
        self.nodes_dict.insert(point=key, data=node)
        return node

    def remove_node_from_dict(self, node):
        for neighbor_coords in node.neighbor_list[1:]:
            neighbor_node = self.nodes_dict.find(neighbor_coords)
            neighbor_node.data.neighbor_list.remove(node.coords.tolist())
        self.nodes_dict.remove(node.coords.tolist())

    def update_graph(self, robot_location, frontiers, updating_map_info, map_info, refresh_frontiers=True):
        node_coords, _ = get_updating_node_coords(robot_location, updating_map_info)

        all_node_list = []
        for coords in node_coords:
            node = self.check_node_exist_in_dict(coords)
            if node is None:
                node = self.add_node_to_dict(coords, frontiers if refresh_frontiers else set(), updating_map_info)
            else:
                node = node.data
            all_node_list.append(node)

        for node in all_node_list:
            if node.need_update_neighbor and np.linalg.norm(node.coords - robot_location) < (
                    self.sensor_range + NODE_RESOLUTION):
                node.update_neighbor_nodes(updating_map_info, self.nodes_dict)

        if refresh_frontiers:
            self.refresh_frontier_cache(map_info)

    def refresh_frontier_cache(self, map_info):
        """Refresh every node from one global Belief Map version, including zero-utility nodes."""
        self.global_frontiers = frozenset(get_frontier_in_map(map_info))
        self.frontier_version += 1
        for entry in self.nodes_dict.__iter__():
            entry.data.refresh_frontier_cache(self.global_frontiers, map_info, self.frontier_version)
        return self.frontier_version

    def get_polar_frontiers(self, node_coords):
        """Return raw counts [N,36,10] and log1p totals in the caller's node order."""
        histograms = np.zeros((len(node_coords), NUM_ANGLES_BIN, NUM_DISTANCE_BINS), dtype=np.int64)
        totals = np.zeros(len(node_coords), dtype=float)
        for i, coords in enumerate(node_coords):
            node = self.nodes_dict.find(tuple(coords)).data
            if node.frontier_version != self.frontier_version:
                raise RuntimeError('Planning requested before the shared frontier refresh')
            histograms[i] = node.frontier_histogram
            totals[i] = node.frontier_count_feature
        return histograms, totals

    def get_all_node_graph(self, robot_location, robot_locations):
        all_node_coords = []
        for node in self.nodes_dict.__iter__():
            all_node_coords.append(node.data.coords)
        all_node_coords = np.array(all_node_coords).reshape(-1, 2)
        utility = []
        frontiers_distribution = []
        highest_utility_angle = []
        heading_visited = []

        n_nodes = all_node_coords.shape[0]
        adjacent_matrix = np.ones((n_nodes, n_nodes)).astype(int)
        coordinate_to_index = {tuple(coords): i for i, coords in enumerate(all_node_coords)}
        for i, coords in enumerate(all_node_coords):
            node = self.nodes_dict.find((coords[0], coords[1])).data
            utility.append(node.utility)
            frontiers_distribution.append(node.frontiers_distribution)
            heading_visited.append(node.heading_visited)
            highest_utility_angle.append(node.highest_utility_angle)

            for neighbor in node.neighbor_list:
                index = coordinate_to_index.get((neighbor[0], neighbor[1]))
                if index is not None:
                    adjacent_matrix[i, index] = 0

        utility = np.array(utility)
        frontiers_distribution = np.array(frontiers_distribution)
        highest_utility_angle = np.array(highest_utility_angle)
        heading_visited = np.array(heading_visited)

        indices = np.argwhere(utility > 0).reshape(-1)
        utility_node_coords = all_node_coords[indices]
        dist_dict, prev_dict = self.Dijkstra(robot_location)
        nearest_utility_coords = robot_location
        nearest_dist = 1e8
        for coords in utility_node_coords:
            dist = dist_dict[(coords[0], coords[1])]
            if 0 < dist < nearest_dist:
                nearest_dist = dist
                nearest_utility_coords = coords

        path_coords, dist = self.a_star(robot_location, nearest_utility_coords)
        guidepost = np.zeros_like(utility)
        for coords in path_coords:
            index = coordinate_to_index[(coords[0], coords[1])]
            guidepost[index] = 1

        robot_in_graph = self.nodes_dict.nearest_neighbors(robot_location.tolist(), 1)[0].data.coords
        current_index = coordinate_to_index[(robot_in_graph[0], robot_in_graph[1])]
        neighbor_indices = np.argwhere(adjacent_matrix[current_index] == 0).reshape(-1)

        occupancy = np.zeros((n_nodes, 1))
        for location in robot_locations:
            location_in_graph = self.nodes_dict.find((location[0], location[1])).data.coords
            index = coordinate_to_index[(location_in_graph[0], location_in_graph[1])]
            if index == current_index:
                occupancy[index] = -1
            else:
                occupancy[index] = 1

        return all_node_coords, utility, guidepost, occupancy, adjacent_matrix, current_index, neighbor_indices, highest_utility_angle, frontiers_distribution, heading_visited, path_coords

    def Dijkstra(self, start):
        return dijkstra_distances(self.nodes_dict, start)

    def get_Dijkstra_path_and_dist(self, dist_dict, prev_dict, end):
        if (end[0], end[1]) not in dist_dict:
            return [], 1e8

        dist = dist_dict[(end[0], end[1])]

        path = [(end[0], end[1])]
        prev_node = prev_dict[(end[0], end[1])]
        while prev_node is not None:
            path.append(prev_node)
            temp = prev_node
            prev_node = prev_dict[temp]

        path.reverse()
        return path[1:], np.round(dist, 2)

    def h(self, coords_1, coords_2):
        h = ((coords_1[0] - coords_2[0]) ** 2 + (coords_1[1] - coords_2[1]) ** 2) ** (1 / 2)
        h = np.round(h, 2)
        return h

    def a_star(self, start, destination, boundary=None, max_dist=None):
        if not self.check_node_exist_in_dict(start):
            print(start)
            Warning("start position is not in node dict")
            return [], 1e8
        if not self.check_node_exist_in_dict(destination):
            Warning("end position is not in node dict")
            return [], 1e8

        if start[0] == destination[0] and start[1] == destination[1]:
            return [destination], 0

        open_list = {(start[0], start[1])}
        closed_list = set()
        g = {(start[0], start[1]): 0}
        parents = {(start[0], start[1]): (start[0], start[1])}

        while len(open_list) > 0:
            n = None
            h_n = 1e8

            for v in open_list:
                h_v = self.h(v, destination)
                if n is not None:
                    node = self.nodes_dict.find(n).data
                    n_coords = node.coords
                    h_n = self.h(n_coords, destination)
                if n is None or g[v] + h_v < g[n] + h_n:
                    n = v
                    node = self.nodes_dict.find(n).data
                    n_coords = node.coords

            if max_dist is not None:
                if g[n] > max_dist:
                    return [], 1e8

            if n_coords[0] == destination[0] and n_coords[1] == destination[1]:
                path = []
                length = g[n]
                while parents[n] != n:
                    path.append(n)
                    n = parents[n]
                path.reverse()
                return path, np.round(length, 2)

            for neighbor_node_coords in node.neighbor_list:
                if self.nodes_dict.find(neighbor_node_coords.tolist()) is None:
                    continue
                if boundary is not None:
                    if not (boundary[0] < neighbor_node_coords[0] < boundary[2] and boundary[1] < neighbor_node_coords[1] < boundary[3]):
                        continue
                cost = ((neighbor_node_coords[0] - n_coords[0]) ** 2 + (
                            neighbor_node_coords[1] - n_coords[1]) ** 2) ** (1 / 2)
                cost = np.round(cost, 2)
                m = (neighbor_node_coords[0], neighbor_node_coords[1])
                if m not in open_list and m not in closed_list:
                    open_list.add(m)
                    parents[m] = n
                    g[m] = g[n] + cost
                else:
                    if g[m] > g[n] + cost:
                        g[m] = g[n] + cost
                        parents[m] = n

                        if m in closed_list:
                            closed_list.remove(m)
                            open_list.add(m)
            open_list.remove(n)
            closed_list.add(n)

        print('Path does not exist!')

        return [], 1e8

class Node:
    def __init__(self, coords, frontiers, updating_map_info, fov, sensor_range, utility_range=None):
        self.coords = coords
        self.utility_range = UTILITY_RANGE if utility_range is None else utility_range
        self.utility = 0
        self.fov = fov
        self.highest_utility_angle = -360
        self.num_angles_bin = NUM_ANGLES_BIN
        self.sensor_range = sensor_range
        self.frontiers_distribution = np.zeros(self.num_angles_bin)
        self.heading_visited = np.zeros(self.num_angles_bin)
        self.frontier_version = -1
        self.frontier_histogram = np.zeros((self.num_angles_bin, NUM_DISTANCE_BINS), dtype=np.int64)
        self.frontier_coords = np.empty((0, 2))
        self.frontier_angles = np.empty(0)
        self.frontier_distances = np.empty(0)
        self.frontier_count_feature = 0.0
        self.observable_frontiers = self.initialize_observable_frontiers(frontiers, updating_map_info)
        self.visited = 0

        self.neighbor_matrix = -np.ones((NUM_NODE_NEIGHBORS, NUM_NODE_NEIGHBORS))
        self.neighbor_list = []
        self.neighbor_list.append(self.coords)
        self.need_update_neighbor = True

    def initialize_observable_frontiers(self, frontiers, updating_map_info):
        self.refresh_frontier_cache(frontiers, updating_map_info, version=-1)
        return self.observable_frontiers

    def refresh_frontier_cache(self, frontiers, map_info, version):
        """20 m capability-independent LOS cache. No GT map or old-utility gate."""
        points = np.asarray(sorted(frontiers), dtype=float).reshape(-1, 2)
        distances = np.linalg.norm(points - self.coords, axis=1)
        points = points[distances <= FRONTIER_REPRESENTATION_RANGE]
        visible = [point for point in points if not check_collision(self.coords, point, map_info)]
        self.frontier_coords = np.asarray(visible, dtype=float).reshape(-1, 2)
        delta = self.frontier_coords - self.coords
        self.frontier_distances = np.linalg.norm(delta, axis=1)
        self.frontier_angles = np.degrees(np.arctan2(delta[:, 1], delta[:, 0])) % 360
        self.frontier_histogram = np.zeros((self.num_angles_bin, NUM_DISTANCE_BINS), dtype=np.int64)
        angle_bins = (self.frontier_angles / (360 / self.num_angles_bin)).astype(int)
        distance_bins = np.minimum((self.frontier_distances / (FRONTIER_REPRESENTATION_RANGE / NUM_DISTANCE_BINS)).astype(int), NUM_DISTANCE_BINS - 1)
        np.add.at(self.frontier_histogram, (angle_bins, distance_bins), 1)
        self.frontier_count_feature = float(np.log1p(self.frontier_histogram.sum()))
        self.frontier_version = version
        self._update_legacy_frontier_features()

    def _update_legacy_frontier_features(self):
        """Keep old 36-D network/reward inputs until the network migration stage."""
        selected = self.frontier_distances < self.utility_range
        self.observable_frontiers = set(map(tuple, self.frontier_coords[selected]))
        self.utility = len(self.observable_frontiers)
        self.frontiers_distribution = np.zeros(self.num_angles_bin)
        self.highest_utility_angle = -360
        if self.utility <= MIN_UTILITY:
            self.utility = 0
            self.observable_frontiers = set()
            return
        angles = self.frontier_angles[selected]
        np.add.at(self.frontiers_distribution, (angles / (360 / self.num_angles_bin)).astype(int), 1)
        half_width = int(self.fov * self.num_angles_bin / 720)
        indices = (np.arange(self.num_angles_bin)[:, None] + np.arange(-half_width, half_width + 1)) % self.num_angles_bin
        scores = self.frontiers_distribution[indices].sum(axis=1)
        self.highest_utility_angle = int(np.argsort(-scores)[0]) * (360 / self.num_angles_bin)

    def candidate_heading_scores(self, fov, sensor_range):
        """Exact V1 counts from raw geometry; histogram bins do not decide eligibility."""
        headings = np.arange(self.num_angles_bin) * (360 / self.num_angles_bin)
        difference = (self.frontier_angles[:, None] - headings[None, :] + 180) % 360 - 180
        return ((self.frontier_distances[:, None] <= 0.9 * sensor_range) &
                (np.abs(difference) <= fov / 2)).sum(axis=0)

    def update_neighbor_nodes(self, updating_map_info, nodes_dict):
        for i in range(self.neighbor_matrix.shape[0]):
            for j in range(self.neighbor_matrix.shape[1]):
                if self.neighbor_matrix[i, j] != -1:
                    continue
                else:
                    center_index = self.neighbor_matrix.shape[0] // 2
                    if i == center_index and j == center_index:
                        self.neighbor_matrix[i, j] = 1
                        continue

                    neighbor_coords = np.around(np.array([self.coords[0] + (i - center_index) * NODE_RESOLUTION,
                                                          self.coords[1] + (j - center_index) * NODE_RESOLUTION]), 1)
                    neighbor_node = nodes_dict.find((neighbor_coords[0], neighbor_coords[1]))
                    if neighbor_node is None:
                        continue
                    else:
                        neighbor_node = neighbor_node.data
                        collision = check_collision(self.coords, neighbor_coords, updating_map_info)
                        neighbor_matrix_x = center_index + (center_index - i)
                        neighbor_matrix_y = center_index + (center_index - j)
                        if not collision:
                            self.neighbor_matrix[i, j] = 1
                            self.neighbor_list.append(neighbor_coords)

                            neighbor_node.neighbor_matrix[neighbor_matrix_x, neighbor_matrix_y] = 1
                            neighbor_node.neighbor_list.append(self.coords)

        # Frontier utility is not evidence that all local edges are resolved.
        if not np.any(self.neighbor_matrix == -1):
            self.need_update_neighbor = False

    def update_node_observable_frontiers(self, frontiers, updating_map_info, map_info):
        # Compatibility entry point; LOS always uses the global Belief Map.
        self.refresh_frontier_cache(frontiers, map_info, self.frontier_version)

    def set_visited(self, heading, fov=None):
        self.visited = 1
        if fov is not None:
            bins = np.arange(self.num_angles_bin) * (360 / self.num_angles_bin)
            difference = (bins - heading + 180) % 360 - 180
            self.heading_visited[np.abs(difference) <= fov / 2] = 1
            return
        # Preserve the legacy call for callers without an explicit physical FoV.
        index = int(heading / 360 * self.num_angles_bin) % self.num_angles_bin
        half_fov_size = int((self.fov / 360) * self.num_angles_bin / 2)
        # Use the same inclusive circular window as heading candidates.
        indices = (index + np.arange(-half_fov_size, half_fov_size + 1)) % self.num_angles_bin
        self.heading_visited[indices] = 1
        self.need_update_neighbor = False
