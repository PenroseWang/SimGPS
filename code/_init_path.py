import os.path as osp
import sys


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


this_dir = osp.dirname(__file__)

# Add folders to PYTHONPATH
add_path(osp.join(this_dir, 'system'))
add_path(osp.join(this_dir, 'signal'))
add_path(osp.join(this_dir, 'tracking'))



