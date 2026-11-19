"""Historical CPU fixtures; never writes through a live runtime symlink."""
import hashlib
from pathlib import Path

REPO=Path(__file__).resolve().parents[2]


def populate_runtime(destination,snapshot,overrides=None):
    overrides=overrides or {}
    saved_bases=None
    def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
    for relative,expected in snapshot.items():
        source=Path(overrides.get(relative,REPO/relative))
        if relative not in overrides and sha(source)!=expected:
            if saved_bases is None:
                saved_bases=list((REPO/'run/a100-gpu0123-20261114/proposals').rglob('*.base'))
            matches=[path for path in saved_bases if sha(path)==expected]
            assert matches,f'Historical fixture bytes unavailable: {relative}'
            source=matches[0]
        path=Path(destination)/relative;path.parent.mkdir(parents=True,exist_ok=True)
        if relative in overrides:path.write_bytes(source.read_bytes())
        else:path.symlink_to(source)
