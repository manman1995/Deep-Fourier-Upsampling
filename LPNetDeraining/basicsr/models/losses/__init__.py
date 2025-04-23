# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
from .losses import (L1Loss, MSELoss, PSNRLoss, SSIMLoss)

__all__ = [
    'L1Loss', 'MSELoss', 'PSNRLoss','SSIMLoss',
]
