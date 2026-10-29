"""V1 process-level failure reporting; no environment or learner semantics."""
import os
import traceback


def episode_map_path(map_directory, episode):
    # Match Env.import_ground_truth exactly; intentionally do not sort here.
    try:
        names = os.listdir(map_directory)
        return os.path.join(map_directory, names[episode % len(names)])
    except (OSError, ZeroDivisionError):
        return None


def episode_error_info(error, worker_id, episode, map_directory):
    return dict(error=True, error_type=type(error).__name__, error_message=str(error),
                traceback=traceback.format_exc(), id=worker_id, meta_agent_id=worker_id,
                episode_number=episode, map_path=episode_map_path(map_directory, episode))


def atomic_checkpoint(payload, path):
    """Publish a complete checkpoint or retain the preceding complete file."""
    from pathlib import Path
    import tempfile
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
