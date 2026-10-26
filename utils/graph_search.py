"""Sparse shortest-distance calculation; guide paths still use the original A*."""
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra


def dijkstra_distances(nodes_dict, start):
    nodes = [entry.data for entry in nodes_dict]
    coords = np.asarray([node.coords for node in nodes], dtype=float).reshape(-1, 2)
    keys = [tuple(point) for point in coords]
    coordinate_to_index = {key: i for i, key in enumerate(keys)}
    start_key = (start[0], start[1])
    assert start_key in coordinate_to_index

    rows, columns = [], []
    for row, node in enumerate(nodes):
        # Repeated neighbor entries represent one edge, not additive weights.
        seen = set()
        for neighbor in node.neighbor_list:
            column = coordinate_to_index.get((neighbor[0], neighbor[1]))
            if column is not None and column not in seen:
                rows.append(row)
                columns.append(column)
                seen.add(column)
    rows = np.asarray(rows, dtype=np.int32)
    columns = np.asarray(columns, dtype=np.int32)
    delta = coords[rows] - coords[columns]
    weights = np.round(np.sqrt(np.sum(delta * delta, axis=1)), 2)
    graph = csr_matrix((weights, (rows, columns)), shape=(len(nodes), len(nodes)))
    distances, predecessors = dijkstra(graph, directed=True, indices=coordinate_to_index[start_key],
                                       return_predecessors=True)
    reachable = np.isfinite(distances) & (distances < 1e8)
    dist_dict = {key: float(distances[i]) if reachable[i] else 1e8 for i, key in enumerate(keys)}
    prev_dict = {key: keys[int(predecessors[i])] if reachable[i] and predecessors[i] >= 0 else None
                 for i, key in enumerate(keys)}
    return dist_dict, prev_dict
