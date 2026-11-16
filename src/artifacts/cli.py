"""Explicit CLI; all extraction and analysis commands in this scaffold use CPU."""

import argparse
import json
from pathlib import Path

from .store import ArtifactStore, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", default="artifacts")
    commands = parser.add_subparsers(dest="command", required=True)
    inventory = commands.add_parser("inventory")
    inventory.add_argument("--registry", default="configs/data_registry.json")
    inventory.add_argument("--root")
    inventory.add_argument("--output")
    inventory.add_argument("--samples", type=int, default=1, help="Number of session headers to inspect per dataset")
    register = commands.add_parser("register-checkpoint")
    register.add_argument("path")
    register.add_argument("--provenance", required=True, help="JSON declaring training data/run provenance, or explicitly unknown")
    register.add_argument("--label", default="")
    extract = commands.add_parser("extract")
    extract.add_argument("checkpoint_id")
    extract.add_argument("--config", required=True)
    extract.add_argument("--registry", default="configs/data_registry.json")
    extract.add_argument("--root")
    extract.add_argument("--label", default="")
    means = commands.add_parser("derive-means")
    means.add_argument("collection_id")
    means.add_argument("--windows", type=int, nargs="+")
    means.add_argument("--label", default="")
    analysis = commands.add_parser("analyze")
    analysis.add_argument("collection_ids", nargs="+")
    analysis.add_argument("--config", required=True)
    analysis.add_argument("--label", default="")
    listing = commands.add_parser("list")
    listing.add_argument("--kind", choices=("checkpoints", "collections", "analyses"))
    inspect = commands.add_parser("show")
    inspect.add_argument("kind", choices=("checkpoints", "collections", "analyses"))
    inspect.add_argument("id")
    args = parser.parse_args()
    store = ArtifactStore(args.store)
    if args.command == "inventory":
        from src.datasets.allen_sessions import AllenRegistry
        if args.samples < 1:
            parser.error("--samples must be positive")
        registry = AllenRegistry(args.registry, args.root)
        result = {}
        for dataset in registry.config["datasets"]:
            sessions = registry.sessions(dataset)
            examples = []
            for session in sessions[:args.samples]:
                description = registry.describe(dataset, session)
                examples.append({"session_id": session, "subject_id": description["subject_id"],
                                 "units": len(description["rows"]), "spikes": description["num_spikes"],
                                 "intervals": description["intervals"],
                                 "probes": len({r["probe_id"] for r in description["rows"]})})
            result[dataset] = {"directory": str(registry.directory(dataset)), "sessions": len(sessions),
                               "spec": registry.spec(dataset), "examples": examples}
        if args.output:
            write_json(args.output, result)
            result = {"output": args.output, "datasets": {
                key: {"sessions": value["sessions"], "headers_checked": len(value["examples"])}
                for key, value in result.items()}}
    elif args.command == "register-checkpoint":
        result = {"checkpoint_id": store.register_checkpoint(args.path, json.loads(Path(args.provenance).read_text()), args.label)}
    elif args.command == "extract":
        from src.datasets.allen_sessions import AllenRegistry
        from .collections import extract
        result = {"collection_id": extract(store, args.checkpoint_id, AllenRegistry(args.registry, args.root),
                                             json.loads(Path(args.config).read_text()), args.label)}
    elif args.command == "derive-means":
        from .collections import derive_means
        result = {"collection_id": derive_means(store, args.collection_id, args.windows, args.label)}
    elif args.command == "analyze":
        from .analyses import analyze
        result = {"analysis_id": analyze(store, args.collection_ids, json.loads(Path(args.config).read_text()), args.label)}
    elif args.command == "show":
        result = store.read(args.kind, args.id)
    else:
        result = [{"kind": m["kind"], "id": m["id"], "label": m["label"], "parents": m["parents"]}
                  for m in store.list(args.kind)]
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
