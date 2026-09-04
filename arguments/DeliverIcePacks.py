import os
import runpy
from pathlib import Path

_SCENE = 'DeliverIcePacks'
_requested = os.environ.get('ED3DGS_SCENE')
if _requested not in (None, '', _SCENE):
    raise ValueError(
        f'{Path(__file__).name} requires ED3DGS_SCENE={_SCENE!r}, got {_requested!r}')
os.environ['ED3DGS_SCENE'] = _SCENE
_shared = runpy.run_path(str(Path(__file__).with_name('MeetingRoom.py')))
for _name, _value in _shared.items():
    if not _name.startswith('__'):
        globals()[_name] = _value
