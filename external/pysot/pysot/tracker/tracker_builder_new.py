# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from external.pysot.pysot.core import cfg_new as cfg
from external.pysot.pysot.tracker.siamrpn_tracker import SiamRPNTracker
from external.pysot.pysot.tracker.siammask_tracker import SiamMaskTracker
from external.pysot.pysot import SiamRPNLTTracker

TRACKS = {
          'SiamRPNTracker': SiamRPNTracker,
          'SiamMaskTracker': SiamMaskTracker,
          'SiamRPNLTTracker': SiamRPNLTTracker
         }


def build_tracker(model):
    return TRACKS[cfg.TRACK.TYPE](model)
