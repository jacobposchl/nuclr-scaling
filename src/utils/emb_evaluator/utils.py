def kfold_assignment(ids: list[str], folds: int) -> list[int]:
    import hashlib

    def deterministic_hash(x: str) -> int:
        return int(hashlib.sha256(x.encode("utf-8")).hexdigest(), 16)

    return [deterministic_hash(x) % folds for x in ids]
