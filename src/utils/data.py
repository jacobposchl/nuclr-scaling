import numpy as np

from torch_brain.data import Data


def apply_unit_mask_(data: Data, mask):
    num_units_before = len(data.units)
    data.units = data.units.select_by_mask(mask)
    num_units_after = len(data.units)

    old_unit_idx = np.arange(num_units_before)[mask]
    unit_idx_remap = np.zeros(num_units_before, dtype=int)
    unit_idx_remap[old_unit_idx] = np.arange(num_units_after)

    spike_mask = np.isin(data.spikes.unit_index, old_unit_idx)
    data.spikes = data.spikes.select_by_mask(spike_mask)
    data.spikes.unit_index = unit_idx_remap[data.spikes.unit_index]

    return data
