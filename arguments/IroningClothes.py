import os

_base_ = './MeetingRoom.py'

_SCENE = 'IroningClothes'
_requested = os.environ.get('ED3DGS_SCENE')
if _requested != _SCENE:
    raise ValueError(
        f'{_SCENE}.py requires ED3DGS_SCENE={_SCENE!r}, '
        f'got {_requested!r}')
