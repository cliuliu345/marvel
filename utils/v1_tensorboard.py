"""Driver-only V1 TensorBoard scalars and import of existing JSON log lines.

Episode tags use the episode number; SAC tags use optimizer_updates, including
the counter restored from a checkpoint. Use a separate directory for each run.
Historical logs without a field leave that curve absent instead of inventing data.

Import without training:
    python -m utils.v1_tensorboard --import-json-log RUN.log --tensorboard-logdir EVENTS
"""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path


UPDATE_FIELDS = ('actor_loss', 'q1_loss', 'q2_loss', 'alpha_loss', 'alpha', 'entropy',
                 'actor_grad_norm', 'q1_grad_norm', 'q2_grad_norm', 'alpha_grad_norm')
EPISODE_FIELDS = ('explored_rate', 'success_rate', 'total_reward', 'macro_steps',
                  'terminated', 'truncated')


@contextmanager
def tensorboard_writer(logdir):
    if logdir is None:
        yield None
        return
    # No writer or TensorBoard dependency is initialized in a Ray Worker.
    from torch.utils.tensorboard import SummaryWriter
    # SummaryWriter.__exit__ flushes and closes, including on exceptions.
    with SummaryWriter(log_dir=str(logdir), flush_secs=10, filename_suffix='.v1') as writer:
        yield writer


def write_update(writer, record):
    if writer is None:
        return
    step = record['optimizer_updates']
    for field in UPDATE_FIELDS:
        if field in record:
            writer.add_scalar('V1/SAC/' + field, record[field], global_step=step)


def write_episode(writer, record):
    if writer is None:
        return
    step = record['episode']
    for field in EPISODE_FIELDS:
        if field in record:
            writer.add_scalar('V1/Episode/' + field, record[field], global_step=step)
    for agent_id, value in enumerate(record.get('agent_returns', ())):
        writer.add_scalar(f'V1/Episode/agent_{agent_id}_return', value, global_step=step)
    reason = record.get('termination_reason')
    if reason is None:
        if record.get('terminated'):
            reason = 'success'
        elif record.get('truncated'):
            reason = 'time_limit'
    if reason is not None:
        writer.add_text('V1/Episode/termination_reason', reason, global_step=step)
    writer.flush()


def import_json_log(path, logdir):
    """Read V1_UPDATE/V1_EPISODE lines; never infer missing rewards or step IDs."""
    path, logdir = Path(path), Path(logdir)
    if any(logdir.glob('events.out.tfevents.*')):
        raise ValueError('Import requires a fresh event directory to avoid duplicate curves')
    counts = {'updates': 0, 'episodes': 0}
    with path.open(encoding='utf-8-sig') as source, tensorboard_writer(logdir) as writer:
        for number, line in enumerate(source, 1):
            line = line.lstrip()
            if line.startswith('V1_UPDATE '):
                kind, emit = 'updates', write_update
            elif line.startswith('V1_EPISODE '):
                kind, emit = 'episodes', write_episode
            else:
                continue
            try:
                record = json.loads(line.split(' ', 1)[1])
                emit(writer, record)
            except (ValueError, TypeError, KeyError) as error:
                raise ValueError(f'{path}:{number}: invalid V1 log record: {error}') from error
            counts[kind] += 1
        if not any(counts.values()):
            raise ValueError(f'{path}: no V1_UPDATE or V1_EPISODE records found')
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--import-json-log', required=True, type=Path)
    parser.add_argument('--tensorboard-logdir', required=True, type=Path)
    args = parser.parse_args()
    counts = import_json_log(args.import_json_log, args.tensorboard_logdir)
    print(json.dumps(dict(logdir=str(args.tensorboard_logdir), **counts)))


if __name__ == '__main__':
    main()
