#!/usr/bin/env python3
"""Route one reviewed queue to its existing fixed four-GPU pool launcher."""
import argparse
import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CAMPAIGN = REPO / 'run/a100-gpu0123-20261114'
POOLS = {
    'gpu0123': (REPO / 'scripts/a100_gpu0123/run_campaign.py', CAMPAIGN),
    'gpu4567': (REPO / 'scripts/a100_gpu4567/run_campaign.py', CAMPAIGN / 'pool-gpu4567'),
}


def check_task_pool(path, pool):
    """The main root contains other pools; ancestry alone cannot select a GPU pool."""
    data = json.loads(path.read_text())
    tasks = data.get('tasks')
    if not isinstance(tasks, list) or not tasks:
        raise ValueError('A nonempty task list is required')
    root = POOLS[pool][1].resolve()
    forbidden = [(CAMPAIGN / name).resolve() for name in ('pool-gpu4567', 'aux-gpu67', 'temp-gpu45')]
    for task in tasks:
        if not isinstance(task, dict):
            raise ValueError('Each task must be an object')
        physical = task.get('physical_pool_id')
        if (pool == 'gpu4567' and physical != 'gpu4567-numa1') or (
                pool == 'gpu0123' and physical not in (None, 'gpu0123-numa0')):
            raise ValueError('Task physical_pool_id differs from --pool')
        argv = task.get('argv')
        if not isinstance(argv, list) or not all(isinstance(x, str) for x in argv):
            raise ValueError('Task argv must be a string list')
        positions = [i for i, x in enumerate(argv) if x == '--out' or x.startswith('--out=')]
        if len(positions) != 1:
            raise ValueError('Each task must have exactly one --out')
        index = positions[0]
        if '--extra' in argv and index > argv.index('--extra'):
            raise ValueError('Task --out cannot be hidden in --extra')
        if argv[index] == '--out':
            if index + 1 == len(argv) or argv[index + 1].startswith('--'):
                raise ValueError('Task --out lacks a value')
            value = argv[index + 1]
        else:
            value = argv[index].split('=', 1)[1]
        if not value:
            raise ValueError('Task --out is empty')
        output = Path(value)
        output = (output if output.is_absolute() else REPO / output).resolve()
        if root not in output.parents:
            raise ValueError('Task output is outside the selected pool root')
        if pool == 'gpu0123' and any(p == output or p in output.parents for p in forbidden):
            raise ValueError('Main pool cannot execute a task in another GPU pool namespace')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--pool', choices=tuple(POOLS), required=True)
    parser.add_argument('--tasks', type=Path, required=True)
    parser.add_argument('--queue-id', required=True)
    parser.add_argument('--environment', type=Path)
    parser.add_argument('--freeze', type=Path)
    parser.add_argument('--continue-independent', action='store_true')
    parser.add_argument('--mooncake-transport', choices=('default', 'nvlink'), default='default')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    try:
        check_task_pool(args.tasks, args.pool)
    except (OSError, ValueError, TypeError, AttributeError) as error:
        parser.error(str(error))
    runner, root = POOLS[args.pool]
    python = str(REPO / '.venv/bin/python')
    command = [python, str(runner), '--root', str(root), '--tasks', str(args.tasks.absolute()),
               '--queue-id', args.queue_id, '--mooncake-transport', args.mooncake_transport]
    for name in ('environment', 'freeze'):
        value = getattr(args, name)
        if value is not None:
            command += ['--' + name, str(value.absolute())]
    for name in ('continue_independent', 'dry_run'):
        if getattr(args, name):
            command.append('--' + name.replace('_', '-'))
    os.execv(python, command)


if __name__ == '__main__':
    main()
