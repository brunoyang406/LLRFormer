from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path as osp
import sys


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


this_dir = osp.dirname(__file__)

# Add project root directory to path
root_path = osp.abspath(osp.join(this_dir, '..'))
add_path(root_path)

# Add lib directory to path
lib_path = osp.abspath(osp.join(this_dir, '..', 'lib'))
add_path(lib_path)

# Add poseeval path if exists
mm_path = osp.abspath(osp.join(this_dir, '..', 'lib/poseeval/py-motmetrics'))
if osp.exists(mm_path):
    add_path(mm_path)
