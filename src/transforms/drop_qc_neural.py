from torch_brain.data import Data
from src.utils.data import apply_unit_mask_


class DropQCNeuralFailProbes:
    """Drop probes that have qc_neural == FAIL"""

    def __call__(self, data) -> Data:
        if not hasattr(data.units, "qc_neural"):
            return data
        mask = data.units.qc_neural.astype(str) != "FAIL"
        return apply_unit_mask_(data, mask)
