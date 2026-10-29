from types import SimpleNamespace as NS
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from utils.v1_worker import V1Worker


@pytest.mark.parametrize('replacement,nonadjacent,stay', [([2., 0.], 1, 1), ([2., 1.], 0, 0), ([2., 2.], 1, 0)])
def test_original_nearest_25_conflict_result(replacement, nonadjacent, stay):
    worker = V1Worker.__new__(V1Worker)
    worker.perf_metrics = dict(conflict_replacements=0, conflict_nonadjacent=0, conflict_stay=0)
    nearest = Mock(return_value=[NS(data=NS(coords=np.array(c))) for c in [[1., 0.], replacement]])
    worker.robot_list = [NS(node_manager=NS(nodes_dict=NS(nearest_neighbors=nearest))) for _ in range(4)]
    table = NS(node_coordinates=torch.tensor([[[1., 0.], [2., 1.]]]), valid_mask=torch.tensor([[True, True]]))
    context = NS(agent_ids=torch.arange(4), current_coordinates=torch.tensor([[0., 0.], [2., 0.], [10., 0.], [20., 0.]]),
                 commanded_coordinates=torch.tensor([[1., 0.], [1., 0.], [11., 0.], [21., 0.]]), action_tables=(table,) * 4)
    actual = worker.resolve_goals(context)
    # Original MARVEL takes the first unoccupied nearest-25 node, even if it
    # is the current position or lies outside the Actor's legal action table.
    np.testing.assert_array_equal(actual, [[1., 0.], replacement, [11., 0.], [21., 0.]])
    nearest.assert_called_once_with([1., 0.], 25)
    assert worker.perf_metrics == dict(conflict_replacements=1, conflict_nonadjacent=nonadjacent, conflict_stay=stay)
