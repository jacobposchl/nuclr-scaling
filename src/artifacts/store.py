"""Content-addressed manifests with atomic publication and payload validation.

No database service is required. Only directories containing a valid manifest
are catalogued. Producers work in private staging directories and never mutate
published artifacts. JSON identity excludes display labels and timestamps.
"""

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import tempfile


KINDS = {"checkpoints", "collections", "analyses"}
SCHEMA_VERSION = 1
PROJECT = Path(__file__).resolve().parents[2]


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_digest(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def implementation_stamp():
    """Fingerprint source and numeric-library versions, including uncommitted edits."""
    paths = sorted((PROJECT / "src").rglob("*.py"))
    packages = {}
    for name in ("numpy", "torch", "torch-brain", "h5py", "scikit-learn", "matplotlib", "umap-learn", "numba", "pynndescent", "joblib"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"source_sha256": digest({str(p.relative_to(PROJECT)): file_digest(p) for p in paths}),
            "packages": packages}


class ArtifactStore:
    def __init__(self, root="artifacts"):
        self.root = Path(root).expanduser().resolve()

    def path(self, kind, artifact_id):
        if kind not in KINDS or not re.fullmatch(r"[0-9a-f]{64}", artifact_id):
            raise ValueError("Expected a valid artifact kind and full SHA256 ID")
        return self.root / kind / artifact_id

    def identity(self, kind, recipe):
        if kind not in KINDS:
            raise ValueError(kind)
        return digest({"schema_version": SCHEMA_VERSION, "kind": kind, "recipe": recipe})

    def read(self, kind, artifact_id, verify=True):
        folder = self.path(kind, artifact_id)
        manifest = json.loads((folder / "manifest.json").read_text())
        if (manifest["id"] != artifact_id or manifest["kind"] != kind
                or manifest["schema_version"] != SCHEMA_VERSION
                or self.identity(kind, manifest["recipe"]) != artifact_id):
            raise ValueError(f"Invalid manifest: {folder}")
        if verify:
            for relative, checksum in manifest["files"].items():
                path = (folder / relative).resolve()
                if not path.is_relative_to(folder) or not path.is_file() or file_digest(path) != checksum:
                    raise ValueError(f"Missing or corrupt artifact payload: {relative}")
        return manifest

    def find(self, kind, recipe):
        artifact_id = self.identity(kind, recipe)
        if not self.path(kind, artifact_id).exists():
            return None
        return self.read(kind, artifact_id)

    @contextmanager
    def build(self, kind, recipe, label="", parents=()):
        """Yield (staging path or None on a cache hit, artifact ID).

        Failed producers leave no completed result. Concurrent identical writes
        are serialized with a lock; an abandoned lock is reported, never silently
        stolen. The winner publishes by a same-filesystem directory rename.
        """
        recipe = json.loads(canonical(recipe))
        artifact_id = self.identity(kind, recipe)
        destination = self.path(kind, artifact_id)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            self.read(kind, artifact_id)
            yield None, artifact_id
            return
        lock = destination.parent / ("." + artifact_id + ".lock")
        try:
            lock.mkdir()
        except FileExistsError:
            raise RuntimeError(f"Artifact is being written (or has a stale lock): {lock}") from None
        stage = None
        try:
            if destination.exists():
                self.read(kind, artifact_id)
                yield None, artifact_id
                return
            stage = Path(tempfile.mkdtemp(prefix=".partial-", dir=destination.parent))
            yield stage, artifact_id
            files = {}
            for path in sorted(stage.rglob("*")):
                if path.is_symlink():
                    raise ValueError("Artifact payloads must not contain symlinks")
                if path.is_file():
                    if path.name == "manifest.json":
                        raise ValueError("manifest.json is reserved for the store")
                    files[str(path.relative_to(stage))] = file_digest(path)
            write_json(stage / "manifest.json", {
                "schema_version": SCHEMA_VERSION, "kind": kind, "id": artifact_id,
                "label": label, "created_at": datetime.now(timezone.utc).isoformat(),
                "recipe": recipe, "parents": list(parents), "files": files,
            })
            os.rename(stage, destination)
            stage = None
        finally:
            if stage is not None:
                shutil.rmtree(stage)
            lock.rmdir()

    def list(self, kind=None):
        kinds = [kind] if kind else sorted(KINDS)
        if any(k not in KINDS for k in kinds):
            raise ValueError(kind)
        for current in kinds:
            for path in sorted((self.root / current).glob("*/manifest.json")):
                if not path.parent.name.startswith("."):
                    yield self.read(current, path.parent.name, verify=False)

    def register_checkpoint(self, path, training_provenance, label=""):
        """Record, but do not copy, a checkpoint. Its bytes are rechecked on use.

        training_provenance should contain the run identifier, dataset/session
        selections, and split artifact, or explicitly state unknown provenance.
        """
        path = Path(path).expanduser().resolve(strict=True)
        recipe = {"checkpoint_sha256": file_digest(path), "training": training_provenance}
        with self.build("checkpoints", recipe, label) as (stage, artifact_id):
            if stage is not None:
                write_json(stage / "reference.json", {"path": str(path), "size": path.stat().st_size})
        return artifact_id

    def checkpoint_path(self, artifact_id):
        manifest = self.read("checkpoints", artifact_id)
        reference = json.loads((self.path("checkpoints", artifact_id) / "reference.json").read_text())
        path = Path(reference["path"])
        if file_digest(path) != manifest["recipe"]["checkpoint_sha256"]:
            raise ValueError("Checkpoint bytes changed after registration")
        return path
