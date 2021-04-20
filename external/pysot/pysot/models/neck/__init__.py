# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from external.pysot.pysot.models.neck.neck import AdjustLayer, AdjustAllLayer

NECKS = {
         'AdjustLayer': AdjustLayer,
         'AdjustAllLayer': AdjustAllLayer
        }

def get_neck(name, **kwargs):
    return NECKS[name](**kwargs)
