"""Wait for a completed local run and resource availability before launching training."""

import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .run_resources import check_free_storage, require_idle_gpu


def queue_pretraining(after_run, config, output, gpu='0', minimum_free_gib=25., poll_seconds=30):
    after_run, output = Path(after_run).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError(f'Continuation output already exists: {output}')
    if not (after_run / 'status.json').is_file():
        raise ValueError(f'Missing predecessor status: {after_run}')
    project = Path(__file__).resolve().parents[2]
    if not (project / 'configs' / (config + '.yaml')).is_file():
        raise ValueError(f'Missing training config: {config}')
    queue_path = output.with_suffix('.queue.json')
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(project / 'scripts/run_allen_pretraining.py'), '--config', config,
               '--output', str(output), '--local', '--require-idle-gpu', '--min-free-gib', str(minimum_free_gib)]
    status = dict(state='waiting_for_predecessor', pid=os.getpid(), after_run=str(after_run),
                  output=str(output), config=config, command=command, gpu=gpu,
                  minimum_free_gib=minimum_free_gib)
    # Exclusive creation prevents two watchers from launching the same output.
    with queue_path.open('x') as stream:
        json.dump(status, stream, indent=2)

    def publish(state, **updates):
        changed = status['state'] != state or any(status.get(k) != v for k, v in updates.items())
        status.update(state=state, **updates, updated_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
        temporary = queue_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(status, indent=2) + '\n')
        os.replace(temporary, queue_path)
        if changed:
            print(json.dumps(status), flush=True)

    try:
        while True:
            predecessor = json.loads((after_run / 'status.json').read_text())
            if predecessor['state'] in {'failed', 'interrupted'}:
                raise RuntimeError(f"Predecessor {predecessor['state']}; continuation was not launched")
            if predecessor['state'] != 'complete':
                time.sleep(poll_seconds)
                continue
            if predecessor['completed_epochs'] != predecessor['target_epochs']:
                raise RuntimeError('Predecessor reports complete without all requested epochs')
            try:
                free_gib = check_free_storage(output, minimum_free_gib)
            except RuntimeError as error:
                publish('waiting_for_storage', reason=str(error))
                time.sleep(poll_seconds)
                continue
            try:
                require_idle_gpu(gpu)
            except RuntimeError as error:
                publish('waiting_for_idle_gpu', reason=str(error))
                time.sleep(poll_seconds)
                continue
            env = os.environ.copy()
            env.update(CUDA_VISIBLE_DEVICES=gpu, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                       MKL_NUM_THREADS='1', NUMBA_NUM_THREADS='1', MPLBACKEND='Agg',
                       PYTHONPATH=str(project), PYTHONUNBUFFERED='1')
            log_path = output.with_suffix('.log')
            with log_path.open('x') as log:
                child = subprocess.Popen(command, cwd=project, env=env, stdin=subprocess.DEVNULL,
                                         stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            publish('running', child_pid=child.pid, log=str(log_path), initial_free_gib=free_gib, reason=None)
            code = child.wait()
            result_path = output / 'status.json'
            result = json.loads(result_path.read_text()) if result_path.exists() else {}
            if code or result.get('state') != 'complete':
                raise RuntimeError(f'Training did not complete (exit {code}); inspect {log_path}')
            publish('complete', completed_epochs=result['completed_epochs'])
            return
    except BaseException as error:
        publish('failed', reason=f'{type(error).__name__}: {error}')
        raise
