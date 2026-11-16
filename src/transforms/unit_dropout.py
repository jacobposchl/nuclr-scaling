import numpy as np
from torch_brain.data import Data
from src.utils.data import apply_unit_mask_


class UnitDropout:
    """A simple unit dropout transform

    It first samples the proportion of units to keep by uniformly sampling
    a float ``p`` in [``min_units``, 1.0]. Then it removes (1 - ``p``) units
    randomly.

    Args:
        min_units: Minimum proportion of units to keep
    """

    def __init__(self, min_units: float):
        assert min_units <= 1.0
        self.min_units = min_units

    def __call__(self, data) -> Data:
        num_units_before = len(data.units)
        min_units = int(num_units_before * self.min_units)

        if num_units_before <= min_units:
            return data

        num_units_after = np.random.randint(min_units, num_units_before + 1)
        sampled_unit_idx = np.random.permutation(num_units_before)[:num_units_after]
        sampled_unit_mask = np.zeros(num_units_before, dtype=bool)
        sampled_unit_mask[sampled_unit_idx] = True

        return apply_unit_mask_(data, sampled_unit_mask)


class FixedUnitDropout:
    """A simple fixed-ratio unit dropout transform

    Args:
        drop_ratio: Ratio of units to drop
    """

    def __init__(self, drop_ratio: float):
        self.drop_ratio = drop_ratio

    def __call__(self, data) -> Data:
        keep_ratio = 1 - self.drop_ratio
        num_units_before = len(data.units)
        keep_units = int(num_units_before * keep_ratio)

        if num_units_before <= keep_units:
            return data

        sampled_unit_idx = np.random.permutation(num_units_before)[:keep_units]
        sampled_unit_mask = np.zeros(num_units_before, dtype=bool)
        sampled_unit_mask[sampled_unit_idx] = True

        return apply_unit_mask_(data, sampled_unit_mask)
