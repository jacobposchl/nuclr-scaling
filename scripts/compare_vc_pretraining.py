"""Extract VC-only embeddings and compare region decoding with saved joint models."""

import argparse
import json
import os
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--vc-run', type=Path, required=True)
    parser.add_argument('--wait-for-run', type=Path, required=True)
    parser.add_argument('--reference-decoding', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--store', default='artifacts')
    parser.add_argument('--atlas-directory', default='results/cache/ibl_atlas_regions')
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()
    for name in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS', 'NUMBA_NUM_THREADS'):
        os.environ[name] = '1'
    os.environ['MPLBACKEND'] = 'Agg'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import torch
    from omegaconf import OmegaConf
    from src.artifacts.collections import extract
    from src.artifacts.store import ArtifactStore, write_json
    from src.artifacts.vc_region_decoding import analyze_vc_region_decoding
    from src.datasets.allen_sessions import AllenRegistry
    from src.utils.run_resources import check_free_storage, require_idle_gpu

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    run = args.vc_run.resolve(strict=True)
    training = json.loads((run / 'status.json').read_text())
    if training['state'] != 'complete' or training['completed_epochs'] != 10:
        raise ValueError('Expected a completed ten-epoch VC-only run')
    checkpoint_path = Path(training['checkpoint_dir']) / 'epoch_10.pt'
    checkpoint_path.resolve(strict=True)
    cfg = OmegaConf.to_container(OmegaConf.load(run / 'resolved_config.yaml'), resolve=True)
    manifest = json.loads((run / 'split_manifest.json').read_text())
    if cfg['model'].get('rate_separation', False) or {r['dataset_id'] for r in manifest['partitions']['pretrain']} != {'allen_visual_coding'}:
        raise ValueError('Expected the original VC-only encoder')
    registry = AllenRegistry(run / 'registry.json')
    store = ArtifactStore(args.store)
    store.read('analyses', args.reference_decoding)
    check_free_storage(args.store, 25.)
    (args.wait_for_run / 'status.json').resolve(strict=True)
    args.output.mkdir(parents=True, exist_ok=False)
    status = dict(state='waiting_for_training', pid=os.getpid(), vc_run=str(run),
                  wait_for_run=str(args.wait_for_run.resolve()), reference_decoding=args.reference_decoding)

    def update(state, **values):
        changed = state != status['state'] or any(status.get(k) != v for k, v in values.items())
        status.update(state=state, updated_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'), **values)
        temporary = args.output / '.status.tmp'
        write_json(temporary, status)
        temporary.replace(args.output / 'status.json')
        if changed:
            print(json.dumps(status), flush=True)

    update('waiting_for_training')
    try:
        while True:
            predecessor = json.loads((args.wait_for_run / 'status.json').read_text())
            if predecessor['state'] in {'failed', 'interrupted'}:
                raise RuntimeError('Preceding training stopped unexpectedly; extraction was not launched')
            if predecessor['state'] != 'complete':
                time.sleep(30)
                continue
            if predecessor['completed_epochs'] != predecessor['target_epochs']:
                raise RuntimeError('Preceding training did not finish all requested epochs')
            try:
                check_free_storage(args.store, 25.)
                require_idle_gpu(args.gpu)
            except RuntimeError as error:
                update('waiting_for_resources', reason=str(error))
                time.sleep(30)
                continue
            break
        checkpoint = store.register_checkpoint(checkpoint_path,
            {'run_id': training['run_id'], 'epoch': 10, 'data_provenance': cfg['data_provenance']}, 'VC-only epoch 10')
        update('extracting', checkpoint_id=checkpoint, reason=None)
        config = {'datasets': [dict(r, split='pretrain') for r in manifest['partitions']['pretrain']],
                  'device': 'cuda', 'fingerprint': 'stat', 'output': 'unit_mean',
                  'step_seconds': cfg['views']['duration'], 'max_windows_per_session': None}
        collection = extract(store, checkpoint, registry, config, 'VC-only epoch 10, full non-opto pooling')
        torch.cuda.empty_cache()
        update('decoding', collection_id=collection)
        analysis = analyze_vc_region_decoding(store, collection, args.reference_decoding, args.atlas_directory)
        update('complete', analysis_id=analysis, result_directory=str(store.path('analyses', analysis)))
    except Exception as error:
        update('failed', error=f'{type(error).__name__}: {error}')
        raise


if __name__ == '__main__':
    main()
