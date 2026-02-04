# Delta calibration support
#
# Copyright (C) 2017-2019  Kevin O'Connor <kevin@koconnor.net>
#
# This file may be distributed under the terms of the GNU GPLv3 license.
import math, logging, collections
import random
import mathutil
from . import probe

#import skspatial
from skspatial.objects import Plane
import numpy as np

# A "stable position" is a 3-tuple containing the number of steps
# taken since hitting the endstop on each delta tower.  Delta
# calibration uses this coordinate system because it allows a position
# to be described independent of the software parameters.

# Load a stable position from a config entry
def load_config_stable(config, option):
    return config.getfloatlist(option, count=3)
 

######################################################################
# Delta calibration object
######################################################################

# The angles and distances of the calibration object found in
# docs/prints/calibrate_size.stl
MeasureAngles = [210., 270., 330., 30., 90., 150.]
MeasureOuterRadius = 65
MeasureRidgeRadius = 5. - .5

# How much to prefer a distance measurement over a height measurement
MEASURE_WEIGHT = 0.5

HexagonProbePattern_37points = [(0.31111, 0.48497), (-0.31111, 0.0), (0.15556, 0.24249), (-0.46667, 0.72746), (-0.46667, -0.72746), (0.62222, 0.0), (0.15556, -0.24249), (-0.62222, 0.0), (0.0, -0.48497), (0.0, 0.96995), (-0.15556, 0.24249), (0.77778, 0.24249), (0.77778, -0.24249), (0.0, 0.48497), (0.0, -0.96995), (0.46667, 0.72746), (-0.15556, -0.24249), (0.46667, -0.72746), (-0.31111, -0.48497), (0.31111, 0.0), (-0.46667, 0.24249), (0.15556, 0.72746), (-0.46667, -0.24249), (-0.31111, 0.48497), (-0.93333, 0.0), (0.62222, -0.48497), (0.15556, -0.72746), (0.62222, 0.48497), (-0.62222, -0.48497), (-0.62222, 0.48497), (0.0, 0.0), (0.46667, 0.24249), (0.31111, -0.48497), (-0.15556, 0.72746), (-0.15556, -0.72746), (0.93333, 0.0), (0.46667, -0.24249), (-0.77778, 0.24249), (-0.77778, -0.24249)]
HexagonProbePattern_61points = [(0.125, 0.20785), (-0.75, -0.41569), (-0.25, 0.41569), (-0.25, -0.83138), (0.25, 0.0), (-0.5, 0.0), (0.75, -0.41569), (-0.625, -0.62354), (0.0, -0.41569), (0.5, 0.41569), (0.375, -0.20785), (0.0, 0.83138), (1.0, 0.0), (0.5, -0.83138), (-0.125, 0.62354), (0.375, 0.20785), (-0.375, -0.20785), (0.125, -0.62354), (-0.375, 0.20785), (-0.75, 0.41569), (0.625, 0.62354), (-0.25, 0.0), (0.25, 0.83138), (0.25, -0.41569), (-0.5, -0.41569), (0.75, 0.41569), (0.0, 0.41569), (-0.5, 0.83138), (0.0, -0.83138), (0.5, 0.0), (-0.625, 0.62354), (-0.125, -0.20785), (-0.875, -0.20785), (0.375, -0.62354), (-0.125, 0.20785), (-0.875, 0.20785), (-0.375, -0.62354), (-1.0, 0.0), (0.125, 0.62354), (0.625, 0.20785), (0.625, -0.20785), (-0.75, 0.0), (-0.25, 0.83138), (-0.375, 0.62354), (-0.25, -0.41569), (0.25, 0.41569), (-0.5, 0.41569), (0.25, -0.83138), (0.75, 0.0), (-0.5, -0.83138), (0.0, 0.0), (-0.625, -0.20785), (0.5, 0.83138), (-0.125, -0.62354), (0.5, -0.41569), (-0.625, 0.20785), (0.875, -0.20785), (0.375, 0.62354), (0.875, 0.20785), (0.125, -0.20785), (0.625, -0.62354)]
HexagonProbePattern_963points = [(0.09375, -0.77942), (0.25, -0.93531), (-0.71875, 0.36373), (-0.40625, -0.05196), (-0.25, -0.83138), (-0.71875, -0.57158), (0.53125, -0.77942), (-0.28125, 0.36373), (0.03125, -0.05196), (-0.28125, -0.57158), (-0.625, 0.31177), (0.375, -0.20785), (0.5, 0.41569), (0.0, 0.51962), (0.15625, -0.15588), (0.46875, -0.05196), (-0.90625, 0.05196), (-0.59375, 0.15588), (0.15625, 0.36373), (-0.8125, 0.20785), (0.1875, -0.31177), (-0.59375, -0.36373), (0.5, -0.51962), (-0.46875, 0.05196), (-0.15625, 0.15588), (-0.15625, -0.36373), (0.15625, -0.57158), (-0.375, 0.20785), (0.625, -0.31177), (0.28125, 0.57158), (-0.03125, 0.05196), (0.28125, -0.36373), (0.0625, 0.20785), (-0.53125, 0.77942), (0.09375, 0.98727), (0.25, 0.83138), (-0.25, 0.93531), (0.21875, 0.25981), (-0.59375, 0.46765), (0.65625, 0.25981), (-0.09375, -0.46765), (0.0625, -0.62354), (-0.90625, -0.25981), (0.34375, -0.46765), (0.5, -0.10392), (-0.46875, 0.6755), (-0.46875, -0.25981), (0.3125, 0.72746), (0.9375, -0.10392), (-0.03125, 0.6755), (0.96875, 0.15588), (0.75, 0.20785), (0.0, 0.0), (-0.34375, 0.88335), (0.0, -0.93531), (0.4375, 0.0), (-0.3125, -0.20785), (-0.3125, -0.72746), (0.875, 0.0), (-0.5, 0.10392), (0.125, -0.20785), (0.125, -0.72746), (-0.09375, -0.15588), (-0.6875, 0.41569), (-0.21875, -0.77942), (-0.0625, 0.10392), (-0.6875, -0.51962), (0.5625, -0.72746), (-0.40625, 0.57158), (-0.25, 0.41569), (0.375, 0.10392), (-0.25, -0.51962), (0.53125, 0.46765), (-0.75, 0.51962), (0.03125, 0.57158), (-0.5625, -0.31177), (0.21875, -0.25981), (0.0, 0.83138), (-0.125, -0.31177), (0.65625, -0.25981), (0.1875, -0.83138), (-0.1875, 0.31177), (-0.1875, -0.62354), (-0.53125, 0.25981), (0.90625, -0.05196), (0.59375, 0.36373), (0.59375, -0.57158), (0.25, 0.31177), (-0.5, -0.41569), (-0.84375, -0.46765), (-0.0625, 0.51962), (-0.6875, -0.10392), (0.6875, 0.31177), (-0.0625, -0.41569), (0.40625, 0.05196), (0.71875, 0.15588), (0.71875, -0.36373), (-0.25, -0.10392), (0.5, 0.20785), (0.375, -0.41569), (0.375, 0.51962), (-0.09375, 0.77942), (0.84375, 0.05196), (-0.4375, 0.72746), (0.1875, 0.93531), (0.34375, 0.77942), (-0.40625, -0.88335), (-0.75, 0.0), (0.03125, -0.88335), (0.59375, 0.57158), (-0.28125, -0.6755), (-0.625, -0.20785), (0.5, -0.62354), (-0.625, -0.72746), (-0.84375, -0.15588), (0.15625, -0.6755), (-0.8125, -0.31177), (-0.375, 0.62354), (-0.21875, 0.46765), (0.40625, 0.6755), (0.0625, 0.62354), (-0.53125, -0.25981), (0.09375, -0.98727), (-0.71875, -0.05196), (0.21875, -0.77942), (-0.59375, 0.36373), (-0.28125, -0.05196), (-0.125, -0.83138), (-0.59375, -0.57158), (-0.9375, 0.31177), (0.1875, 0.41569), (0.8125, 0.10392), (0.1875, -0.51962), (-0.15625, 0.36373), (0.15625, -0.05196), (-0.90625, 0.15588), (-0.90625, -0.36373), (0.625, 0.41569), (-0.3125, 0.51962), (-0.15625, -0.57158), (0.625, -0.51962), (0.28125, -0.15588), (-0.78125, 0.05196), (-0.46875, 0.15588), (0.28125, 0.36373), (-0.46875, -0.36373), (-0.6875, 0.20785), (0.3125, -0.31177), (-0.34375, 0.05196), (-0.03125, 0.15588), (-0.03125, -0.36373), (-0.25, 0.20785), (0.75, -0.31177), (-0.0625, 0.83138), (0.375, 0.83138), (-0.125, 0.93531), (-0.09375, 0.25981), (0.34375, 0.25981), (-0.40625, -0.46765), (-0.25, -0.62354), (-0.46875, 0.46765), (0.53125, -0.57158), (0.03125, -0.46765), (0.8125, 0.51962), (0.1875, -0.10392), (0.8125, -0.41569), (-0.78125, -0.25981), (0.46875, -0.46765), (0.0, 0.72746), (0.625, -0.10392), (-0.34375, 0.6755), (0.65625, 0.15588), (0.4375, 0.20785), (0.4375, 0.72746), (-0.3125, 0.0), (-0.3125, -0.93531), (0.875, 0.20785), (0.125, 0.0), (-0.21875, 0.88335), (0.125, -0.93531), (0.5625, 0.0), (-0.8125, 0.10392), (-0.1875, -0.20785), (-0.1875, -0.72746), (-0.40625, -0.15588), (0.59375, -0.6755), (-0.53125, -0.77942), (-0.375, 0.10392), (0.25, -0.20785), (0.25, -0.72746), (-0.71875, 0.57158), (0.03125, -0.15588), (-0.5625, 0.41569), (0.0625, 0.10392), (-0.5625, -0.51962), (0.21875, 0.46765), (-0.28125, 0.57158), (-0.125, 0.41569), (-0.875, -0.31177), (-0.125, -0.51962), (0.65625, 0.46765), (0.5, 0.62354), (0.15625, 0.57158), (-0.3125, 0.83138), (-0.4375, -0.31177), (0.34375, -0.25981), (0.78125, -0.25981), (0.03125, -0.98727), (0.3125, -0.83138), (-0.5, 0.31177), (-0.5, -0.62354), (-0.84375, 0.25981), (0.59375, -0.05196), (-0.0625, 0.31177), (0.9375, -0.20785), (-0.8125, -0.41569), (0.71875, -0.15588), (0.71875, 0.36373), (0.375, 0.31177), (-0.375, -0.41569), (0.09375, 0.05196), (0.40625, 0.15588), (0.40625, -0.36373), (-0.5625, -0.10392), (0.1875, 0.20785), (0.0625, -0.41569), (0.0625, 0.51962), (-0.40625, 0.77942), (0.53125, 0.05196), (0.84375, -0.36373), (-0.125, -0.10392), (0.03125, 0.77942), (0.96875, 0.05196), (0.3125, 0.93531), (0.46875, 0.77942), (-0.28125, -0.88335), (-0.625, 0.0), (-0.59375, -0.6755), (0.15625, -0.88335), (-0.8125, 0.41569), (-0.9375, -0.20785), (-0.15625, -0.6755), (0.625, -0.62354), (0.28125, -0.6755), (-0.6875, 0.62354), (-0.53125, 0.46765), (0.09375, 0.6755), (0.09375, -0.25981), (-0.25, 0.62354), (0.53125, 0.6755), (0.21875, 0.88335), (-0.09375, -0.77942), (-0.90625, 0.36373), (-0.59375, -0.05196), (-0.4375, -0.83138), (0.34375, -0.77942), (0.5, 0.10392), (-0.46875, 0.36373), (-0.15625, -0.05196), (-0.46875, -0.57158), (0.3125, 0.41569), (0.9375, 0.10392), (-0.8125, -0.10392), (-0.03125, 0.36373), (0.28125, -0.05196), (-0.78125, 0.15588), (-0.78125, -0.36373), (0.75, 0.41569), (-0.1875, 0.51962), (0.0, -0.31177), (0.3125, -0.51962), (0.75, -0.51962), (-0.65625, 0.05196), (-0.34375, 0.15588), (-0.34375, -0.36373), (-0.03125, -0.57158), (-0.5625, 0.20785), (0.4375, -0.31177), (-0.21875, 0.05196), (-0.125, 0.20785), (0.875, -0.31177), (-0.09375, 0.98727), (0.0625, 0.83138), (-1.0, 0.0), (-0.40625, 0.25981), (0.03125, 0.25981), (-0.71875, -0.46765), (0.8125, 0.31177), (-0.78125, 0.46765), (0.46875, 0.25981), (-0.28125, -0.46765), (0.5, 0.51962), (-0.125, -0.62354), (0.5, -0.41569), (0.65625, -0.57158), (0.15625, -0.46765), (-0.3125, 0.72746), (0.3125, -0.10392), (-0.65625, 0.6755), (-0.65625, -0.25981), (0.125, 0.72746), (0.75, -0.10392), (-0.21875, 0.6755), (0.78125, 0.15588), (0.5625, 0.20785), (0.5625, 0.72746), (-0.1875, 0.0), (-0.1875, -0.93531), (0.25, 0.0), (-0.5, -0.20785), (-0.5, -0.72746), (-0.71875, -0.15588), (0.6875, 0.0), (-0.6875, 0.10392), (-0.0625, -0.20785), (-0.0625, -0.72746), (-0.28125, -0.15588), (0.71875, -0.6755), (-0.25, 0.10392), (0.375, -0.72746), (-0.59375, 0.57158), (-0.4375, 0.41569), (0.1875, 0.62354), (-0.4375, -0.51962), (0.34375, 0.46765), (-0.15625, 0.57158), (-0.75, -0.31177), (0.625, 0.62354), (0.78125, 0.46765), (0.03125, -0.25981), (-0.1875, 0.83138), (0.46875, -0.25981), (0.0, -0.83138), (0.15625, -0.98727), (-0.8125, 0.31177), (0.4375, -0.83138), (-0.375, 0.31177), (-0.375, -0.62354), (0.40625, 0.36373), (0.71875, -0.05196), (0.40625, -0.57158), (0.0625, 0.31177), (-0.6875, -0.41569), (0.84375, -0.15588), (0.84375, 0.36373), (0.09375, 0.15588), (0.09375, -0.36373), (-0.875, -0.10392), (-0.25, 0.51962), (-0.25, -0.41569), (0.21875, 0.05196), (0.53125, 0.15588), (0.53125, -0.36373), (-0.4375, -0.10392), (0.3125, 0.20785), (0.6875, -0.51962), (-0.28125, 0.77942), (0.65625, 0.05196), (-0.625, 0.72746), (0.5, 0.83138), (0.0, 0.93531), (0.15625, 0.77942), (-0.9375, 0.0), (-0.15625, -0.88335), (0.90625, 0.25981), (-0.46875, -0.6755), (0.28125, -0.88335), (0.3125, -0.62354), (0.09375, 0.46765), (-0.03125, -0.6755), (0.59375, -0.46765), (0.75, -0.62354), (-0.5625, 0.62354), (0.21875, 0.6755), (1.0, 0.0), (-0.125, 0.62354), (0.65625, 0.6755), (-0.09375, -0.98727), (0.34375, 0.88335), (-0.40625, -0.77942), (-0.90625, -0.05196), (0.03125, -0.77942), (0.1875, 0.10392), (0.8125, -0.20785), (-0.46875, -0.05196), (-0.78125, 0.36373), (0.59375, -0.15588), (0.0, 0.41569), (-0.78125, -0.57158), (0.625, 0.10392), (0.0, -0.51962), (-0.34375, 0.36373), (-0.03125, -0.05196), (-0.5, 0.51962), (-0.34375, -0.57158), (0.4375, 0.41569), (0.46875, -0.77942), (-0.3125, -0.31177), (0.4375, -0.51962), (-0.96875, 0.05196), (-0.65625, 0.15588), (-0.65625, -0.36373), (0.875, 0.41569), (-0.875, 0.20785), (0.125, -0.31177), (0.71875, 0.57158), (-0.53125, 0.05196), (-0.21875, 0.15588), (0.90625, -0.25981), (-0.21875, -0.36373), (-0.4375, 0.20785), (0.5625, -0.31177), (-0.25, 0.83138), (0.03125, 0.98727), (-0.71875, 0.25981), (-0.71875, -0.6755), (-0.28125, 0.25981), (0.5, 0.31177), (0.15625, 0.25981), (-0.59375, -0.46765), (0.9375, 0.31177), (-0.4375, -0.62354), (0.1875, -0.41569), (-0.65625, 0.46765), (-0.15625, -0.46765), (0.625, 0.51962), (0.0, -0.10392), (0.625, -0.41569), (0.78125, -0.57158), (0.28125, -0.46765), (-0.1875, 0.72746), (0.4375, -0.10392), (-0.53125, 0.6755), (0.46875, 0.15588), (0.59375, 0.77942), (0.09375, 0.88335), (0.25, 0.72746), (-0.5, 0.0), (0.875, -0.10392), (0.6875, 0.20785), (-0.0625, 0.0), (-0.8125, -0.20785), (-0.0625, -0.93531), (0.375, 0.0), (-0.375, -0.20785), (-0.375, -0.72746), (-0.59375, -0.15588), (0.40625, -0.6755), (-0.5625, 0.10392), (0.0625, -0.20785), (0.0625, -0.72746), (-0.15625, -0.15588), (-0.75, 0.41569), (-0.125, 0.10392), (-0.75, -0.51962), (0.03125, 0.46765), (-0.46875, 0.57158), (0.3125, 0.62354), (0.46875, 0.46765), (-0.03125, 0.57158), (-0.5, 0.83138), (-0.625, -0.31177), (0.75, 0.62354), (0.15625, -0.25981), (-0.3125, -0.83138), (-0.15625, -0.98727), (0.125, -0.83138), (-0.6875, 0.31177), (-0.6875, -0.62354), (0.09375, 0.36373), (0.40625, -0.05196), (0.09375, -0.57158), (-0.25, 0.31177), (0.75, -0.20785), (0.53125, -0.15588), (0.84375, -0.05196), (0.53125, 0.36373), (-0.5625, -0.41569), (0.96875, -0.15588), (-0.09375, 0.05196), (0.21875, 0.15588), (0.21875, -0.36373), (-0.75, -0.10392), (0.0, 0.20785), (-0.125, -0.41569), (-0.125, 0.51962), (0.65625, 0.57158), (0.34375, 0.05196), (0.65625, -0.36373), (-0.59375, 0.77942), (-0.3125, 0.93531), (-0.15625, 0.77942), (0.78125, 0.05196), (0.125, 0.93531), (0.28125, 0.77942), (0.59375, 0.25981), (-0.03125, -0.88335), (-0.34375, -0.6755), (0.4375, -0.62354), (0.71875, -0.46765), (-0.09375, 0.6755), (-0.09375, -0.25981), (-0.4375, 0.62354), (0.34375, 0.6755), (0.03125, 0.88335), (0.8125, 0.0), (-0.28125, -0.77942), (0.5, -0.20785), (-0.78125, -0.05196), (0.5, -0.72746), (-0.3125, 0.41569), (0.15625, -0.77942), (0.3125, 0.10392), (-0.3125, -0.51962), (-0.65625, 0.36373), (-0.34375, -0.05196), (-0.8125, 0.51962), (-0.65625, -0.57158), (0.125, 0.41569), (0.75, 0.10392), (0.125, -0.51962), (-0.21875, 0.36373), (-0.375, 0.51962), (-0.96875, 0.15588), (-0.21875, -0.57158), (0.5625, 0.41569), (0.40625, 0.57158), (-0.1875, -0.31177), (0.5625, -0.51962), (-0.84375, 0.05196), (-0.53125, 0.15588), (0.59375, -0.25981), (-0.53125, -0.36373), (-0.75, 0.20785), (0.25, -0.31177), (0.6875, -0.31177), (-0.125, 0.83138), (0.15625, 0.98727), (-0.59375, 0.25981), (0.1875, 0.31177), (0.1875, -0.62354), (-0.15625, 0.25981), (0.625, 0.31177), (0.28125, 0.25981), (-0.46875, -0.46765), (0.3125, 0.51962), (-0.3125, -0.10392), (0.3125, -0.41569), (0.46875, -0.57158), (-0.03125, -0.46765), (0.75, 0.51962), (0.125, -0.10392), (-0.5, 0.72746), (0.75, -0.41569), (-0.84375, -0.25981), (-0.0625, 0.72746), (-0.8125, 0.0), (0.5625, -0.10392), (0.375, 0.20785), (0.375, 0.72746), (-0.375, 0.0), (0.40625, -0.88335), (0.0625, 0.0), (-0.6875, -0.20785), (0.0625, -0.93531), (-0.90625, -0.15588), (0.09375, -0.6755), (-0.875, 0.10392), (-0.25, -0.20785), (-0.25, -0.72746), (-0.46875, -0.15588), (0.53125, -0.6755), (-0.4375, 0.10392), (-0.78125, 0.57158), (-0.03125, -0.15588), (-0.625, 0.41569), (0.0, 0.62354), (-0.625, -0.51962), (0.15625, 0.46765), (-0.34375, 0.57158), (-0.9375, -0.31177), (0.4375, 0.62354), (-0.15625, -0.25981), (-0.375, 0.83138), (0.28125, -0.25981), (-0.1875, -0.83138), (-0.03125, -0.98727), (0.59375, -0.77942), (0.25, -0.83138), (0.09375, -0.05196), (-0.5625, 0.31177), (-0.5625, -0.62354), (0.21875, 0.36373), (0.53125, -0.05196), (0.21875, -0.57158), (-0.125, 0.31177), (0.875, -0.20785), (-0.875, -0.41569), (0.65625, -0.15588), (-0.40625, 0.05196), (0.96875, -0.05196), (-0.09375, 0.15588), (-0.09375, -0.36373), (-0.3125, 0.20785), (0.65625, 0.36373), (-0.4375, -0.41569), (-0.4375, 0.51962), (0.03125, 0.05196), (0.34375, 0.15588), (0.34375, -0.36373), (-0.625, -0.10392), (0.125, 0.20785), (-0.46875, 0.77942), (0.78125, 0.57158), (0.46875, 0.05196), (0.78125, -0.36373), (0.3125, 0.83138), (-0.1875, 0.93531), (-0.03125, 0.77942), (0.25, 0.93531), (-0.34375, -0.88335), (0.71875, 0.25981), (-0.65625, -0.6755), (-0.875, 0.41569), (0.125, -0.62354), (-0.09375, 0.46765), (-0.21875, -0.6755), (0.40625, -0.46765), (0.5625, -0.62354), (-0.40625, 0.6755), (-0.40625, -0.25981), (0.84375, -0.46765), (-0.75, 0.62354), (0.03125, 0.6755), (0.8125, 0.20785), (0.46875, 0.6755), (0.5, 0.0), (0.15625, 0.88335), (-0.59375, -0.77942), (0.9375, 0.0), (0.1875, -0.20785), (0.1875, -0.72746), (-0.15625, -0.77942), (0.0, 0.10392), (0.625, -0.20785), (-0.65625, -0.05196), (0.625, -0.72746), (0.40625, -0.15588), (-0.1875, 0.41569), (0.28125, -0.77942), (0.4375, 0.10392), (-0.1875, -0.51962), (-0.53125, 0.36373), (-0.21875, -0.05196), (0.59375, 0.46765), (-0.6875, 0.51962), (0.25, 0.41569), (0.09375, 0.57158), (0.875, 0.10392), (-0.5, -0.31177), (0.25, -0.51962), (-0.53125, -0.57158), (-0.84375, 0.15588), (-0.84375, -0.36373), (0.6875, 0.41569), (0.53125, 0.57158), (-0.0625, -0.31177), (0.71875, -0.25981), (-0.625, 0.20785), (0.375, -0.31177), (-0.4375, 0.83138), (-0.15625, 0.98727), (-0.90625, 0.25981), (-0.46875, 0.25981), (0.3125, 0.31177), (-0.03125, 0.25981), (-0.78125, -0.46765), (0.75, 0.31177), (-0.625, -0.62354), (0.0, -0.41569), (-0.84375, 0.46765), (-0.34375, -0.46765), (0.4375, 0.51962), (-0.1875, -0.10392), (0.4375, -0.41569), (0.90625, 0.05196), (-0.375, 0.72746), (0.25, -0.10392), (0.875, -0.41569), (0.28125, 0.15588), (0.40625, 0.77942), (-0.09375, 0.88335), (0.0625, 0.72746), (-0.6875, 0.0), (0.6875, -0.10392), (0.09375, -0.88335), (-0.25, 0.0), (-0.25, -0.93531), (-0.5625, -0.20785), (-0.5625, -0.72746), (-0.78125, -0.15588), (0.21875, -0.6755), (-0.75, 0.10392), (-0.125, -0.20785), (-0.125, -0.72746), (-0.34375, -0.15588), (0.65625, -0.6755), (-0.3125, 0.62354), (-0.15625, 0.46765), (-0.65625, 0.57158), (0.125, 0.62354), (0.28125, 0.46765), (-0.21875, 0.57158), (0.78125, -0.46765), (0.5625, 0.62354), (-0.03125, -0.25981), (-0.5, -0.83138), (-0.0625, -0.83138), (-0.875, 0.31177), (-0.09375, 0.36373), (0.21875, -0.05196), (0.375, -0.83138), (-0.09375, -0.57158), (-0.4375, 0.31177), (0.5625, -0.20785), (0.1875, 0.51962), (0.34375, -0.15588), (-0.71875, 0.05196), (0.65625, -0.05196), (-0.40625, 0.15588), (0.34375, 0.36373), (-0.40625, -0.36373), (-0.75, -0.41569), (0.34375, -0.57158), (0.78125, -0.15588), (-0.28125, 0.05196), (0.03125, 0.15588), (0.03125, -0.36373), (-0.9375, -0.10392), (-0.1875, 0.20785), (0.8125, -0.31177), (0.78125, 0.36373), (0.46875, 0.57158), (0.15625, 0.05196), (0.46875, -0.36373), (0.25, 0.20785), (-0.34375, 0.77942), (0.4375, 0.83138), (-0.0625, 0.93531), (0.40625, 0.25981), (-0.21875, -0.88335), (-0.40625, 0.46765), (0.84375, 0.25981), (-0.53125, -0.6755), (0.09375, -0.46765), (0.25, -0.62354), (-0.71875, 0.6755), (-0.71875, -0.25981), (0.53125, -0.46765), (0.6875, -0.62354), (-0.28125, 0.6755), (-0.28125, -0.25981), (-0.625, 0.62354), (0.5, 0.72746), (0.15625, 0.6755), (0.9375, 0.20785), (0.1875, 0.0), (-0.15625, 0.88335), (0.1875, -0.93531), (0.625, 0.0), (0.28125, 0.88335), (-0.46875, -0.77942), (-0.3125, 0.10392), (0.3125, -0.20785), (-0.96875, -0.05196), (0.3125, -0.72746), (0.09375, -0.15588), (-0.5, 0.41569), (-0.03125, -0.77942), (0.125, 0.10392), (-0.5, -0.51962), (-0.84375, 0.36373), (-0.53125, -0.05196), (-0.0625, 0.41569), (0.5625, 0.10392), (-0.0625, -0.51962), (0.71875, 0.46765), (-0.5625, 0.51962), (0.21875, 0.57158), (0.375, 0.41569), (-0.375, -0.31177), (0.375, -0.51962), (0.40625, -0.25981), (-0.9375, 0.20785), (0.0625, -0.31177), (0.1875, 0.83138), (0.84375, -0.25981), (-0.03125, 0.98727), (-0.78125, 0.25981), (0.0, 0.31177), (0.0, -0.62354), (-0.34375, 0.25981), (0.4375, 0.31177), (-0.3125, -0.41569), (-0.65625, -0.46765), (0.125, 0.51962), (-0.5, -0.10392), (0.875, 0.31177), (0.125, -0.41569), (0.28125, -0.57158), (0.59375, 0.05196), (0.90625, 0.15588), (0.90625, -0.36373), (-0.0625, -0.10392), (-0.21875, -0.46765), (0.5625, -0.41569), (0.5625, 0.51962), (0.09375, 0.77942), (-0.40625, 0.88335), (-0.25, 0.72746), (0.375, -0.10392), (0.53125, 0.77942), (-0.5625, 0.0), (0.21875, -0.88335), (-0.125, 0.0), (-0.875, -0.20785), (-0.125, -0.93531), (-0.09375, -0.6755), (-0.4375, -0.20785), (-0.4375, -0.72746), (-0.65625, -0.15588), (0.34375, -0.6755), (-0.625, 0.10392), (0.78125, 0.25981), (-0.21875, -0.15588), (-0.1875, 0.62354), (-0.03125, 0.46765), (-0.53125, 0.57158), (0.59375, 0.6755), (0.25, 0.62354), (-0.34375, -0.25981), (0.6875, 0.62354), (-0.375, -0.83138), (0.40625, -0.77942), (-0.40625, 0.36373), (-0.09375, -0.05196), (0.0625, -0.83138), (-0.40625, -0.57158), (-0.75, 0.31177), (-0.75, -0.62354), (0.03125, 0.36373), (0.34375, -0.05196), (-0.71875, 0.15588), (-0.71875, -0.36373), (0.8125, 0.41569), (0.6875, -0.20785), (0.03125, -0.57158), (0.8125, -0.51962), (0.46875, -0.15588), (-0.59375, 0.05196), (0.78125, -0.05196), (-0.28125, 0.15588), (-0.28125, -0.36373), (-0.5, 0.20785), (0.5, -0.31177), (0.46875, 0.36373), (-0.625, -0.41569), (-0.15625, 0.05196), (0.15625, 0.15588), (0.15625, -0.36373), (-0.625, 0.51962), (-0.0625, 0.20785), (0.9375, -0.31177), (0.28125, 0.05196), (0.125, 0.83138), (-0.21875, 0.77942), (0.0625, 0.93531), (0.09375, 0.25981), (-0.71875, 0.46765), (0.53125, 0.25981), (-0.0625, -0.62354), (-0.28125, 0.46765), (0.71875, -0.57158), (0.21875, -0.46765), (0.375, -0.62354), (-0.59375, 0.6755), (-0.59375, -0.25981), (0.65625, -0.46765), (0.1875, 0.72746), (0.8125, -0.10392), (-0.15625, 0.6755), (0.84375, 0.15588), (0.625, 0.20785), (0.625, 0.72746), (0.28125, 0.6755), (0.3125, 0.0), (-0.03125, 0.88335), (0.3125, -0.93531), (0.75, 0.0), (0.0, -0.20785), (0.0, -0.72746), (-0.34375, -0.77942), (-0.1875, 0.10392), (-0.8125, -0.51962), (0.4375, -0.20785), (-0.84375, -0.05196), (0.4375, -0.72746), (0.21875, -0.15588), (-0.375, 0.41569), (0.25, 0.10392), (-0.375, -0.51962), (0.40625, 0.46765), (-0.09375, 0.57158), (0.0625, 0.41569), (0.6875, 0.10392), (-0.6875, -0.31177), (0.84375, 0.46765), (0.0625, -0.51962), (0.34375, 0.57158), (-0.25, -0.31177), (0.53125, -0.25981), (0.5, -0.83138), (-0.3125, 0.31177), (-0.3125, -0.62354), (-0.65625, 0.25981), (0.125, 0.31177), (0.90625, -0.15588), (-0.21875, 0.25981), (0.90625, 0.36373), (0.5625, 0.31177), (-0.1875, -0.41569), (0.59375, 0.15588), (0.59375, -0.36373), (-0.375, -0.10392), (-0.53125, -0.46765), (0.25, -0.41569), (0.25, 0.51962), (0.71875, 0.05196), (-0.5625, 0.72746), (0.6875, 0.51962), (0.0625, -0.10392), (0.6875, -0.41569), (0.21875, 0.77942), (-0.28125, 0.88335), (-0.125, 0.72746), (-0.875, 0.0), (-0.09375, -0.88335), (-0.4375, 0.0), (-0.40625, -0.6755), (0.34375, -0.88335), (-0.75, -0.20785), (-0.96875, -0.15588), (0.03125, -0.6755), (-0.9375, 0.10392), (-0.53125, -0.15588), (0.46875, -0.6755), (-0.5, 0.62354), (-0.34375, 0.46765), (-0.0625, 0.62354), (0.71875, 0.6755), (0.375, 0.62354), (0.40625, 0.88335), (-0.21875, -0.25981)]

# Convert distance measurements made on the calibration object to
# 3-tuples of (actual_distance, stable_position1, stable_position2)
def measurements_to_distances(measured_params, delta_params):
    # Extract params
    mp = measured_params
    dp = delta_params
    scale = mp['SCALE'][0]
    cpw = mp['CENTER_PILLAR_WIDTHS']
    center_widths = [cpw[0], cpw[2], cpw[1], cpw[0], cpw[2], cpw[1]]
    center_dists = [od - cw
                    for od, cw in zip(mp['CENTER_DISTS'], center_widths)]
    outer_dists = [
        od - opw
        for od, opw in zip(mp['OUTER_DISTS'], mp['OUTER_PILLAR_WIDTHS']) ]
    # Convert angles in degrees to an XY multiplier
    obj_angles = list(map(math.radians, MeasureAngles))
    xy_angles = list(zip(map(math.cos, obj_angles), map(math.sin, obj_angles)))
    # Calculate stable positions for center measurements
    inner_ridge = MeasureRidgeRadius * scale
    inner_pos = [(ax * inner_ridge, ay * inner_ridge, 0.)
                 for ax, ay in xy_angles]
    outer_ridge = (MeasureOuterRadius + MeasureRidgeRadius) * scale
    outer_pos = [(ax * outer_ridge, ay * outer_ridge, 0.)
                 for ax, ay in xy_angles]
    center_positions = [
        (cd, dp.calc_stable_position(ip), dp.calc_stable_position(op))
        for cd, ip, op in zip(center_dists, inner_pos, outer_pos)]
    # Calculate positions of outer measurements
    outer_center = MeasureOuterRadius * scale
    start_pos = [(ax * outer_center, ay * outer_center) for ax, ay in xy_angles]
    shifted_angles = xy_angles[2:] + xy_angles[:2]
    first_pos = [(ax * inner_ridge + spx, ay * inner_ridge + spy, 0.)
                 for (ax, ay), (spx, spy) in zip(shifted_angles, start_pos)]
    second_pos = [(ax * outer_ridge + spx, ay * outer_ridge + spy, 0.)
                  for (ax, ay), (spx, spy) in zip(shifted_angles, start_pos)]
    outer_positions = [
        (od, dp.calc_stable_position(fp), dp.calc_stable_position(sp))
        for od, fp, sp in zip(outer_dists, first_pos, second_pos)]
    return center_positions + outer_positions


######################################################################
# Delta Calibrate class
######################################################################

class DeltaCalibrate:
    def __init__(self, config):
        self.printer = config.get_printer()
        self.printer.register_event_handler("klippy:connect",
                                            self.handle_connect)
        # # # Calculate probing points
        # # # 37 point probe pattern:
        radius = config.getfloat('radius', above=0.)
 
        # points = [(0., 0.)]
        # dist = radius 
        # for i in range(12):
        #     r = math.radians(90. + 30. * i)
        #     points.append((math.cos(r) * dist, math.sin(r) * dist))
        
        # dist = radius * 0.7
        # for i in range(12):
        #     r = math.radians(90. + 30. * i + 15.)
        #     points.append((math.cos(r) * dist, math.sin(r) * dist))

        # dist = radius * 0.3
        # for i in range(12):
        #     r = math.radians(90. + 30. * i)
        #     points.append((math.cos(r) * dist, math.sin(r) * dist))
        
        points = [(x * radius, y * radius) for x, y in HexagonProbePattern_61points]

        self.probe_helper = probe.ProbePointsHelper(
            config, self.probe_finalize, default_points=points)
        self.probe_helper.minimum_points(3)
        
        # Restore probe stable positions
        self.last_probe_positions = []
        for i in range(999):
            height = config.getfloat("height%d" % (i,), None)
            if height is None:
                break
            height_pos = load_config_stable(config, "height%d_pos" % (i,))
            self.last_probe_positions.append((height, height_pos))
        # Restore manually entered heights
        self.manual_heights = []
        for i in range(999):
            height = config.getfloat("manual_height%d" % (i,), None)
            if height is None:
                break
            height_pos = load_config_stable(config, "manual_height%d_pos"
                                            % (i,))
            self.manual_heights.append((height, height_pos))
        # Restore distance measurements
        self.delta_analyze_entry = {'SCALE': (1.,)}
        self.last_distances = []
        for i in range(999):
            dist = config.getfloat("distance%d" % (i,), None)
            if dist is None:
                break
            distance_pos1 = load_config_stable(config, "distance%d_pos1" % (i,))
            distance_pos2 = load_config_stable(config, "distance%d_pos2" % (i,))
            self.last_distances.append((dist, distance_pos1, distance_pos2))
        # Register gcode commands
        self.gcode = self.printer.lookup_object('gcode')
        self.gcode.register_command('DELTA_CALIBRATE', self.cmd_DELTA_CALIBRATE,
                                    desc=self.cmd_DELTA_CALIBRATE_help)
        self.gcode.register_command('DELTA_ANALYZE', self.cmd_DELTA_ANALYZE,
                                    desc=self.cmd_DELTA_ANALYZE_help)
    def handle_connect(self):
        kin = self.printer.lookup_object('toolhead').get_kinematics()
        if not hasattr(kin, "get_calibration"):
            raise self.printer.config_error(
                "Delta calibrate is only for delta printers")
    def save_state(self, probe_positions, distances, delta_params):
        # Save main delta parameters
        configfile = self.printer.lookup_object('configfile')
        delta_params.save_state(configfile)
        # Save probe stable positions
        section = 'delta_calibrate'
        configfile.remove_section(section)
        for i, (z_offset, spos) in enumerate(probe_positions):
            configfile.set(section, "height%d" % (i,), z_offset)
            configfile.set(section, "height%d_pos" % (i,),
                           "%.3f,%.3f,%.3f" % tuple(spos))
        # Save manually entered heights
        for i, (z_offset, spos) in enumerate(self.manual_heights):
            configfile.set(section, "manual_height%d" % (i,), z_offset)
            configfile.set(section, "manual_height%d_pos" % (i,),
                           "%.3f,%.3f,%.3f" % tuple(spos))
        # Save distance measurements
        for i, (dist, spos1, spos2) in enumerate(distances):
            configfile.set(section, "distance%d" % (i,), dist)
            configfile.set(section, "distance%d_pos1" % (i,),
                           "%.3f,%.3f,%.3f" % tuple(spos1))
            configfile.set(section, "distance%d_pos2" % (i,),
                           "%.3f,%.3f,%.3f" % tuple(spos2))
    def probe_finalize(self, offsets, positions):
        # Convert positions into (z_offset, stable_position) pairs
        z_offset = offsets[2]
        kin = self.printer.lookup_object('toolhead').get_kinematics()
        delta_params = kin.get_calibration()
        probe_positions = [(z_offset, delta_params.calc_stable_position(p))
                           for p in positions]
        # Perform analysis
        self.calculate_params(probe_positions, self.last_distances)
    def calculate_params(self, probe_positions, distances):
        height_positions = self.manual_heights + probe_positions
        # Setup for coordinate descent analysis
        kin = self.printer.lookup_object('toolhead').get_kinematics()
        orig_delta_params = odp = kin.get_calibration()
        adj_params, params = odp.coordinate_descent_params(distances)
        logging.info("Calculating delta_calibrate with:\n%s\n%s\n"
                     "Initial delta_calibrate parameters: %s",
                     height_positions, distances, params)
        z_weight = 0.5
        if distances:
            z_weight = len(distances) / (MEASURE_WEIGHT * len(probe_positions))
        

        
        
        # Perform coordinate descent
        def delta_errorfunc(params):
            try:
                # Build new delta_params for params under test
                delta_params = orig_delta_params.new_calibration(params)
                getpos = delta_params.get_position_from_stable
                
                arms = [arm for arm in delta_params.arms]
                arm2 = [arm**2 for arm in delta_params.arms]
                arm_r =[r for r in delta_params.radii]   
                             
                # Calculate z height errors
                total_error = 0.
                
                #Error metric for height is average and stddev of z deviations
                z_errors = []
                for z_offset, stable_pos in height_positions:
                    x, y, z = getpos(stable_pos)
                    # z += x * params['tilt_x']*.001 + y * params['tilt_y']*.001
                    z_errors.append((z - z_offset)**2)
                
                error_height= 100 *sum(z_errors) #/ len(z_errors)
                total_error += error_height
 
                z_errors=[]
                # Calculate distance errors
                if len(distances) == 0:
                    return 9999999999999.9
                for dist, stable_pos1, stable_pos2 in distances:
                    x1, y1, z1 = getpos(stable_pos1)
                    x2, y2, z2 = getpos(stable_pos2)
                    d = math.sqrt((x1-x2)**2 + (y1-y2)**2 )
                    z_errors.append((d - dist)**2)
                
                error_distance = 100 * sum(z_errors) #/ len(z_errors)
                total_error += error_distance

                return total_error
            except ValueError:
                return 9999999999999.9
        new_params = mathutil.background_coordinate_descent(
            self.printer, adj_params, params, delta_errorfunc)
        # Log and report results
        logging.info("Calculated delta_calibrate parameters: %s", new_params)
        new_delta_params = orig_delta_params.new_calibration(new_params)
        for z_offset, spos in height_positions:
            logging.info("height orig: %.6f new: %.6f goal: %.6f",
                         orig_delta_params.get_position_from_stable(spos)[2],
                         new_delta_params.get_position_from_stable(spos)[2],
                         z_offset)
        for dist, spos1, spos2 in distances:
            x1, y1, z1 = orig_delta_params.get_position_from_stable(spos1)
            x2, y2, z2 = orig_delta_params.get_position_from_stable(spos2)
            orig_dist = math.sqrt((x1-x2)**2 + (y1-y2)**2)
            x1, y1, z1 = new_delta_params.get_position_from_stable(spos1)
            x2, y2, z2 = new_delta_params.get_position_from_stable(spos2)
            new_dist = math.sqrt((x1-x2)**2 + (y1-y2)**2)
            logging.info("distance orig: %.6f new: %.6f goal: %.6f",
                         orig_dist, new_dist, dist)
        # Store results for SAVE_CONFIG
        self.save_state(probe_positions, distances, new_delta_params)
        self.gcode.respond_info(
            "The SAVE_CONFIG command will update the printer config file\n"
            "with these parameters and restart the printer.")
    cmd_DELTA_CALIBRATE_help = "Delta calibration script"
    def cmd_DELTA_CALIBRATE(self, gcmd):
        self.probe_helper.start_probe(gcmd)
    def add_manual_height(self, height):
        # Determine current location of toolhead
        toolhead = self.printer.lookup_object('toolhead')
        toolhead.flush_step_generation()
        kin = toolhead.get_kinematics()
        kin_spos = {s.get_name(): s.get_commanded_position()
                    for s in kin.get_steppers()}
        kin_pos = kin.calc_position(kin_spos)
        # Convert location to a stable position
        delta_params = kin.get_calibration()
        stable_pos = tuple(delta_params.calc_stable_position(kin_pos))
        # Add to list of manual heights
        self.manual_heights.append((height, stable_pos))
        self.gcode.respond_info(
            "Adding manual height: %.3f,%.3f,%.3f is actually z=%.3f"
            % (kin_pos[0], kin_pos[1], kin_pos[2], height))
    def do_extended_calibration(self):
        # Extract distance positions
        if len(self.delta_analyze_entry) <= 1:
            distances = self.last_distances
        elif len(self.delta_analyze_entry) < 5:
            raise self.gcode.error("Not all measurements provided")
        else:
            kin = self.printer.lookup_object('toolhead').get_kinematics()
            delta_params = kin.get_calibration()
            distances = measurements_to_distances(
                self.delta_analyze_entry, delta_params)
        if not self.last_probe_positions:
            raise self.gcode.error(
                "Must run basic calibration with DELTA_CALIBRATE first")
        # Perform analysis
        self.calculate_params(self.last_probe_positions, distances)
    cmd_DELTA_ANALYZE_help = "Extended delta calibration tool"
    def cmd_DELTA_ANALYZE(self, gcmd):
        # Check for manual height entry
        mheight = gcmd.get_float('MANUAL_HEIGHT', None)
        if mheight is not None:
            self.add_manual_height(mheight)
            return
        # Parse distance measurements
        args = {'CENTER_DISTS': 6, 'CENTER_PILLAR_WIDTHS': 3,
                'OUTER_DISTS': 6, 'OUTER_PILLAR_WIDTHS': 6, 'SCALE': 1}
        for name, count in args.items():
            data = gcmd.get(name, None)
            if data is None:
                continue
            try:
                parts = list(map(float, data.split(',')))
            except:
                raise gcmd.error("Unable to parse parameter '%s'" % (name,))
            if len(parts) != count:
                raise gcmd.error("Parameter '%s' must have %d values"
                                 % (name, count))
            self.delta_analyze_entry[name] = parts
            logging.info("DELTA_ANALYZE %s = %s", name, parts)
        # Perform analysis if requested
        action = gcmd.get('CALIBRATE', None)
        if action is not None:
            if action != 'extended':
                raise gcmd.error("Unknown calibrate action")
            self.do_extended_calibration()

def load_config(config):
    return DeltaCalibrate(config)

######################################################################
# error metric calculations, none accounted for endstops so the
# trilateration uses same gemoetry as the delta kinematics
######################################################################

                # #When each tower height is equal to the elevation formed by arm and radius
                # #the nozzle should be touching the bed.
                # sphere_coords = [(t[0], t[1], math.sqrt(a**2 - r**2)) for t, a, r in zip(delta_params.towers, delta_params.arms, delta_params.radii)]
                # even_tower_pos = mathutil.trilateration(sphere_coords, arm2)
                # mid_err = even_tower_pos[2]**2
                # total_error += 25 * (even_tower_pos[2]**2 )

                # #Calculate distances between each tower
                # tower_dist_a_c= math.sqrt( (delta_params.towers[0][0]-delta_params.towers[2][0])**2 + (delta_params.towers[0][1]-delta_params.towers[2][1])**2 )
                # tower_dist_b_c= math.sqrt( (delta_params.towers[1][0]-delta_params.towers[2][0])**2 + (delta_params.towers[1][1]-delta_params.towers[2][1])**2 )
                # tower_dist_a_b= math.sqrt( (delta_params.towers[0][0]-delta_params.towers[1][0])**2 + (delta_params.towers[0][1]-delta_params.towers[1][1])**2 )

                # #Calculate cartesian position with a arm vertical
                # sphere_coords =  [(delta_params.towers[0][0], delta_params.towers[0][1], arms[0]),
                #                   (delta_params.towers[1][0], delta_params.towers[1][1], math.sqrt(arm2[1]-tower_dist_a_b**2)),
                #                   (delta_params.towers[2][0], delta_params.towers[2][1], math.sqrt(arm2[2]-tower_dist_a_c**2))]
                # arm_vertical_pos = mathutil.trilateration(sphere_coords, arm2)
                # a_error = ((arm_vertical_pos[0]-delta_params.towers[0][0])**2 + (arm_vertical_pos[1]-delta_params.towers[0][1])**2 + arm_vertical_pos[2]**2)
                # total_error += 25 * ((arm_vertical_pos[0]-delta_params.towers[0][0])**2 + (arm_vertical_pos[1]-delta_params.towers[0][1])**2 + arm_vertical_pos[2]**2)
                
                # #calculate cartesian position with b arm vertical
                # sphere_coords =  [ (delta_params.towers[0][0], delta_params.towers[0][1], math.sqrt(arm2[0]-tower_dist_a_b**2)),
                #                    (delta_params.towers[1][0], delta_params.towers[1][1], arms[1]),
                #                    (delta_params.towers[2][0], delta_params.towers[2][1], math.sqrt(arm2[2]-tower_dist_b_c**2))]
                # arm_vertical_pos = mathutil.trilateration(sphere_coords, arm2)
                # b_error = ((arm_vertical_pos[0]-delta_params.towers[1][0])**2 + (arm_vertical_pos[1]-delta_params.towers[1][1])**2 + arm_vertical_pos[2]**2)
                # total_error += 25 * ((arm_vertical_pos[0]-delta_params.towers[1][0])**2 + (arm_vertical_pos[1]-delta_params.towers[1][1])**2 + arm_vertical_pos[2]**2)
                
                # #calculate cartesian position with c arm vertical
                # sphere_coords =  [ (delta_params.towers[0][0], delta_params.towers[0][1], math.sqrt(arm2[0]-tower_dist_a_c**2)),
                #                    (delta_params.towers[1][0], delta_params.towers[1][1], math.sqrt(arm2[1]-tower_dist_b_c**2)),
                #                    (delta_params.towers[2][0], delta_params.towers[2][1], arms[2])]
                # arm_vertical_pos = mathutil.trilateration(sphere_coords, arm2)
                # c_error = ((arm_vertical_pos[0]-delta_params.towers[2][0])**2 + (arm_vertical_pos[1]-delta_params.towers[2][1])**2 + arm_vertical_pos[2]**2)
                # total_error += 25 * ((arm_vertical_pos[0]-delta_params.towers[2][0])**2 + (arm_vertical_pos[1]-delta_params.towers[2][1])**2 + arm_vertical_pos[2]**2)
                
                # # Constrain endstop and arm length changes so they are not independent
                # # Find tower heights based only on absolute endstop
                # lowest_endstop = min(delta_params.endstops)
                # endstop_offsets = [e - lowest_endstop for e in delta_params.endstops]
                # sphere_coords = [(t[0], t[1], ae-lowest_endstop-eo) for t, ae, eo in zip(delta_params.towers, delta_params.abs_endstops, endstop_offsets)]
                # even_tower_pos = mathutil.trilateration(sphere_coords, arm2)
                
                # mid_err = 100 * (even_tower_pos[0]**2 + even_tower_pos[1]**2 + even_tower_pos[2]**2)
                # total_error += mid_err *0.0
                
                # #same problem as the other methods calculations aren't independent 
                               
                # # #Constrain arm lengths to be roughly the same distance
                # arm_error = 100 * ((arms[0] - arms[1])**2 + (arms[1] - arms[2])**2 + (arms[2] - arms[0])**2)
                # total_error += arm_error *0.0
                
                # # #constrain radii to be roughly the same
                # radius_error = 100 * ((arm_r[0] - arm_r[1])**2 + (arm_r[1] - arm_r[2])**2 + (arm_r[2] - arm_r[0])**2)
                # total_error += radius_error *0.0
                                               