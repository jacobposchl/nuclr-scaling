"""Queue an explicitly authorized local run after another run finishes."""

import argparse
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--after-run', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--min-free-gib', type=float, default=25.)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.utils.pretraining_queue import queue_pretraining
    queue_pretraining(args.after_run, args.config, args.output, args.gpu, args.min_free_gib)


if __name__ == '__main__':
    main()
