# ------------------------------------------------------------------------------
# Loss Functions
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .joints_loss import (
    JointsMSELoss,
    JointsWeightedMSELoss,
    JointsOHKMMSELoss
)

__all__ = [
    'JointsMSELoss',
    'JointsWeightedMSELoss',
    'JointsOHKMMSELoss',
]

