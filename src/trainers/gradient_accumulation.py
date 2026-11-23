"""Explicit batch sizing and loss-weighted gradient accumulation."""


def pretraining_batch_settings(cfg, world_size):
    """Return the per-GPU micro-batch size and number of accumulation steps."""
    batch_size = cfg.batch_size
    micro_batch_size = cfg.get("micro_batch_size")
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if micro_batch_size is None:
        if batch_size % world_size:
            raise ValueError("batch_size must be divisible by world_size")
        return batch_size // world_size, 1
    if type(micro_batch_size) is not int or micro_batch_size <= 0:
        raise ValueError("micro_batch_size must be a positive integer or null")
    global_micro_batch = micro_batch_size * world_size
    if batch_size % global_micro_batch:
        raise ValueError(
            f"batch_size ({batch_size}) must be divisible by micro_batch_size "
            f"({micro_batch_size}) * world_size ({world_size}) = {global_micro_batch}; "
            "batch sizes are never rounded"
        )
    return micro_batch_size, batch_size // global_micro_batch


class GradientAccumulator:
    """Accumulate weighted loss sums, then normalize before the optimizer step.

    NuCLR supplies a detached normalization weight shared across ranks. DDP can
    average each micro-batch's gradients as usual; the final normalization still
    reproduces the loss over the combined window pairs.
    """

    def __init__(self, optimizer, *, weighted):
        self.optimizer = optimizer
        self.weighted = weighted
        self.weight = 0.0
        self.logs = {}
        self.optimizer.zero_grad()

    def backward(self, loss, logs):
        weight = logs["train/loss_weight"] if self.weighted else 1.0
        if weight == 0:
            return
        (loss * weight).backward()
        self.weight += weight
        for key, value in logs.items():
            contribution = value if key == "train/loss_weight" else value * weight
            self.logs[key] = self.logs.get(key, 0.0) + contribution

    def step(self):
        if self.weight == 0:
            return None
        if self.weighted:
            for group in self.optimizer.param_groups:
                for parameter in group["params"]:
                    if parameter.grad is not None:
                        parameter.grad.div_(self.weight)
        logs = {key: value if key == "train/loss_weight" else value / self.weight
                for key, value in self.logs.items()}
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.weight = 0.0
        self.logs = {}
        return logs
