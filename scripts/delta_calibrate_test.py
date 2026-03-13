#!/usr/bin/env python3
# Delta calibration performance testing tool
#
# Generates simulated probe height measurements and distance measurement
# stable positions from a [delta_true_calibration] config section,
# saves them to the [delta_calibrate] SAVE_CONFIG block, and optionally
# runs the calibration optimization to mimic DELTA_CALIBRATE / DELTA_ANALYZE.
#
# Usage: delta_calibrate_test.py [options] <config_file>
#
# Copyright (C) 2024  Kevin O'Connor <kevin@koconnor.net>
#
# This file may be distributed under the terms of the GNU GPLv3 license.
from __future__ import print_function
import argparse, configparser, io, math, os, random, sys
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# Calibration object geometry constants (matching delta_calibrate.py)
MEASURE_ANGLES = [210., 270., 330., 30., 90., 150.]
MEASURE_OUTER_RADIUS = 65.
MEASURE_RIDGE_RADIUS = 5. - .5  # 4.5 mm

# Normalized coordinates are multiplied by probe_radius to
# get the actual probe positions.
HexagonProbePattern_39points = [
    (0.31111, 0.48497), (-0.31111, 0.0), (0.15556, 0.24249),
    (-0.46667, 0.72746), (-0.46667, -0.72746), (0.62222, 0.0),
    (0.15556, -0.24249), (-0.62222, 0.0), (0.0, -0.48497),
    (0.0, 0.96995), (-0.15556, 0.24249), (0.77778, 0.24249),
    (0.77778, -0.24249), (0.0, 0.48497), (0.0, -0.96995),
    (0.46667, 0.72746), (-0.15556, -0.24249), (0.46667, -0.72746),
    (-0.31111, -0.48497), (0.31111, 0.0), (-0.46667, 0.24249),
    (0.15556, 0.72746), (-0.46667, -0.24249), (-0.31111, 0.48497),
    (-0.93333, 0.0), (0.62222, -0.48497), (0.15556, -0.72746),
    (0.62222, 0.48497), (-0.62222, -0.48497), (-0.62222, 0.48497),
    (0.0, 0.0), (0.46667, 0.24249), (0.31111, -0.48497),
    (-0.15556, 0.72746), (-0.15556, -0.72746), (0.93333, 0.0),
    (0.46667, -0.24249), (-0.77778, 0.24249), (-0.77778, -0.24249),
]

SAVE_CONFIG_HEADER = (
    "#*# <---------------------- SAVE_CONFIG ---------------------->")
SAVE_CONFIG_WARNING = (
    "#*# DO NOT EDIT THIS BLOCK OR BELOW. The contents are auto-generated.")


######################################################################
# Standalone delta calibration geometry
######################################################################

class TrueDeltaCalibration:
    """Delta calibration parameters representing the physical ground truth.

    Mirrors the DeltaCalibration class from klippy/kinematics/delta.py but
    is self-contained so the CLI tool has no klipper runtime dependency.
    """
    def __init__(self, radius, angles, arms, endstops, stepdists,
                 bed_tilt_x=0., bed_tilt_y=0.):
        self.radius = radius
        self.angles = angles
        self.arms = arms
        self.endstops = endstops
        self.stepdists = stepdists
        self.bed_tilt_x = bed_tilt_x
        self.bed_tilt_y = bed_tilt_y
        # Tower XY positions in the cartesian plane
        radian_angles = [math.radians(a) for a in angles]
        self.towers = [(math.cos(a) * radius, math.sin(a) * radius)
                       for a in radian_angles]
        # Absolute Z height of each tower endstop (top of travel)
        radius2 = radius ** 2
        self.abs_endstops = [e + math.sqrt(a**2 - radius2)
                             for e, a in zip(endstops, arms)]

    def bed_z(self, x, y):
        """Return the bed surface height at (x, y) given the bed tilt."""
        return self.bed_tilt_x * x + self.bed_tilt_y * y

    def calc_stable_position(self, coord):
        """Convert a cartesian (x, y, z) coordinate to a stable stepper
        position (3-tuple of steps taken since hitting the endstop).

        Mirrors DeltaCalibration.calc_stable_position in delta.py.
        """
        x, y, z = coord
        steppos = [
            math.sqrt(a**2 - (t[0] - x)**2 - (t[1] - y)**2) + z
            for t, a in zip(self.towers, self.arms)
        ]
        return [
            (ep - sp) / sd
            for sd, ep, sp in zip(self.stepdists, self.abs_endstops, steppos)
        ]

    def get_position_from_stable(self, stable_position):
        """Convert a stable stepper position to cartesian (x, y, z) coordinates.

        Mirrors DeltaCalibrationParams.get_position_from_stable.
        """
        sphere_coords = [
            (t[0], t[1], es - sp * sd)
            for sd, t, es, sp in zip(self.stepdists, self.towers,
                                     self.abs_endstops, stable_position)
        ]
        return trilateration(sphere_coords, [a**2 for a in self.arms])


######################################################################
# Standalone math utilities for delta calibration
######################################################################

def _mat_dot(m1, m2):
    return m1[0]*m2[0] + m1[1]*m2[1] + m1[2]*m2[2]

def _mat_magsq(m1):
    return m1[0]**2 + m1[1]**2 + m1[2]**2

def _mat_add(m1, m2):
    return [m1[0]+m2[0], m1[1]+m2[1], m1[2]+m2[2]]

def _mat_sub(m1, m2):
    return [m1[0]-m2[0], m1[1]-m2[1], m1[2]-m2[2]]

def _mat_mul(m1, s):
    return [m1[0]*s, m1[1]*s, m1[2]*s]

def _mat_cross(m1, m2):
    return [m1[1]*m2[2] - m1[2]*m2[1],
            m1[2]*m2[0] - m1[0]*m2[2],
            m1[0]*m2[1] - m1[1]*m2[0]]

def trilateration(sphere_coords, radius2):
    """Find the intersection of three spheres (trilateration).

    sphere_coords: list of 3 sphere center (x, y, z) tuples
    radius2:       list of 3 squared radii

    Mirrors mathutil.trilateration from the klipper codebase.
    """
    sc1, sc2, sc3 = sphere_coords
    s21 = _mat_sub(sc2, sc1)
    s31 = _mat_sub(sc3, sc1)
    d = math.sqrt(_mat_magsq(s21))
    ex = _mat_mul(s21, 1. / d)
    i = _mat_dot(ex, s31)
    vect_ey = _mat_sub(s31, _mat_mul(ex, i))
    ey = _mat_mul(vect_ey, 1. / math.sqrt(_mat_magsq(vect_ey)))
    ez = _mat_cross(ex, ey)
    j = _mat_dot(ey, s31)
    x = (radius2[0] - radius2[1] + d**2) / (2. * d)
    y = (radius2[0] - radius2[2] - x**2 + (x - i)**2 + j**2) / (2. * j)
    z = -math.sqrt(radius2[0] - x**2 - y**2)
    return _mat_add(sc1, _mat_add(_mat_mul(ex, x),
                    _mat_add(_mat_mul(ey, y), _mat_mul(ez, z))))


######################################################################
# Delta calibration parameters (for optimization)
######################################################################

class DeltaCalibrationParams:
    """Delta printer calibration parameters used in the optimization loop.

    Mirrors DeltaCalibration from klippy/kinematics/delta.py, extended to
    include bed_tilt parameters so they can be jointly optimized with the
    delta geometry during coordinate descent.
    """
    def __init__(self, radius, angles, arms, endstops, stepdists,
                 bed_tilt_x=0., bed_tilt_y=0.):
        self.radius = radius
        self.angle_error = [a - x for a, x in zip(angles, [210., 330., 90.])]
        offset = sum(self.angle_error) -min(self.angle_error) - max(self.angle_error)
        self.angles = [a - offset for a in angles]
        self.arms = arms
        self.endstops = endstops
        self.stepdists = stepdists
        self.bed_tilt_x = bed_tilt_x
        self.bed_tilt_y = bed_tilt_y
        radian_angles = [math.radians(a) for a in angles]
        self.towers = [(math.cos(a) * radius, math.sin(a) * radius)
                       for a in radian_angles]
        radius2 = radius ** 2
        self.abs_endstops = [e + math.sqrt(a**2 - radius2)
                             for e, a in zip(endstops, arms)]

    def get_position_from_stable(self, stable_position):
        """Convert a stable stepper position to cartesian (x, y, z) coordinates.

        Mirrors DeltaCalibration.get_position_from_stable from delta.py.
        """
        sphere_coords = [
            (t[0], t[1], es - sp * sd)
            for sd, t, es, sp in zip(self.stepdists, self.towers,
                                     self.abs_endstops, stable_position)
        ]
        return trilateration(sphere_coords, [a**2 for a in self.arms])
    
    def get_position_from_tower(self, tower_positions):
        """Convert tower carriage positions to cartesian (x, y, z) coordinates.

        """
        sphere_coords = [
            (t[0], t[1], tp )
            for tp, t in zip(tower_positions, self.towers)
        ]
        return trilateration(sphere_coords, [a**2 for a in self.arms])

    def calc_stable_position(self, coord):
        """Convert a cartesian (x, y, z) coordinate to a stable stepper
        position (3-tuple of steps taken since hitting the endstop).

        Mirrors TrueDeltaCalibration.calc_stable_position.
        """
        x, y, z = coord
        steppos = [
            math.sqrt(a**2 - (t[0] - x)**2 - (t[1] - y)**2) + z
            for t, a in zip(self.towers, self.arms)
        ]
        return [
            (ep - sp) / sd
            for sd, ep, sp in zip(self.stepdists, self.abs_endstops, steppos)
        ]
        
    def calc_tower_from_position(self, coord):
        """Return the three tower carriage heights for a given (x, y, z)."""
        x, y, z = coord
        return [
            math.sqrt(a**2 - (t[0] - x)**2 - (t[1] - y)**2) + z
            for t, a in zip(self.towers, self.arms)
        ]

    def coordinate_descent_params(self, is_extended):
        """Return (adj_params, params) for coordinate descent optimization.

        adj_params: tuple of parameter names to adjust during descent
        params:     dict of current parameter values

        bed_tilt_x and bed_tilt_y are always included as adjustment parameters.
        With is_extended=True (distance measurements available), arm lengths
        and tower angles are also included, matching DELTA_ANALYZE behavior.
        """
        adj_params = ('radius',
                      'endstop_a', 'endstop_b', 'endstop_c' 
                        , 'bed_tilt_x', 'bed_tilt_y'
                      )
        if is_extended:
            adj_params += ('arm_a', 'arm_b', 'arm_c',
                           'angle_a', 'angle_b', 'angle_c')
        params = {
            'radius': self.radius,
            'bed_tilt_x': self.bed_tilt_x,
            'bed_tilt_y': self.bed_tilt_y,
        }
        for i, axis in enumerate('abc'):
            params['angle_' + axis] = self.angles[i]
            params['arm_' + axis] = self.arms[i]
            params['endstop_' + axis] = self.endstops[i]
            params['stepdist_' + axis] = self.stepdists[i]
        return adj_params, params

    def new_calibration(self, params):
        """Create a new DeltaCalibrationParams from a coordinate descent
        params dict."""
        radius = params['radius']
        angles = [params['angle_' + a] for a in 'abc']
        arms = [params['arm_' + a] for a in 'abc']
        endstops = [params['endstop_' + a] for a in 'abc']
        stepdists = [params['stepdist_' + a] for a in 'abc']
        bed_tilt_x = params.get('bed_tilt_x', self.bed_tilt_x)
        bed_tilt_y = params.get('bed_tilt_y', self.bed_tilt_y)
        return DeltaCalibrationParams(radius, angles, arms, endstops, stepdists,
                                      bed_tilt_x, bed_tilt_y)


######################################################################
# Configuration file parsing
######################################################################

def _parse_save_config_block(lines):
    """Extract lines from the SAVE_CONFIG block (stripping the #*# prefix).

    Returns a list of plain text lines from the block, or an empty list if
    the block is not present.  Preamble lines (header and warning text)
    that appear before the first INI section are discarded.
    """
    save_config_lines = []
    in_block = False
    found_section = False
    for line in lines:
        if line.strip() == SAVE_CONFIG_HEADER.strip():
            in_block = True
            continue
        if in_block:
            # Each line in the block starts with "#*#"
            if line.startswith('#*#'):
                content = line[3:]
                # Strip a single leading space if present
                if content.startswith(' '):
                    content = content[1:]
                # Skip preamble (warning / blank lines before first section)
                if not found_section:
                    stripped = content.strip()
                    if stripped.startswith('[') and stripped.endswith(']'):
                        found_section = True
                    else:
                        # Not a section header -- skip this preamble line
                        continue
                save_config_lines.append(content)
            else:
                # Non-comment line signals end of block (shouldn't happen in
                # well-formed configs, but handle gracefully)
                break
    return save_config_lines


def parse_klipper_config(filename):
    """Parse a klipper printer.cfg, returning a configparser.ConfigParser.

    Both the main config and the SAVE_CONFIG block are merged so that
    SAVE_CONFIG values (which override the main config in klipper) are
    visible to the caller.
    """
    with open(filename, 'r') as f:
        lines = f.readlines()

    # Collect main config lines (everything before the SAVE_CONFIG block)
    main_lines = []
    save_config_start = None
    for i, line in enumerate(lines):
        if line.strip() == SAVE_CONFIG_HEADER.strip():
            save_config_start = i
            break
        main_lines.append(line)

    # Parse main config
    config = configparser.ConfigParser(strict=False, interpolation=None)
    config.read_string(''.join(main_lines))

    # Parse SAVE_CONFIG block and merge into config (overrides main config)
    if save_config_start is not None:
        save_lines = _parse_save_config_block(lines[save_config_start:])
        config.read_string(''.join(save_lines))

    return config, lines, save_config_start


def load_true_calibration(config):
    """Load a TrueDeltaCalibration from the [delta_true_calibration] section.

    Required parameters:
      delta_radius      - radial distance from center to tower positions (mm)
      endstop_a         - position_endstop for tower A (mm)
      arm_length        - arm length shared by all towers, OR per-tower
                          arm_length_a / arm_length_b / arm_length_c (mm)

    Optional parameters (defaults shown):
      endstop_b/c       - defaults to endstop_a
      arm_length_b/c    - defaults to arm_length_a (or arm_length)
      angle_a           - 210.0 (degrees)
      angle_b           - 330.0 (degrees)
      angle_c           - 90.0 (degrees)
      stepdist_a        - 0.0001 (mm/step)
      stepdist_b/c      - defaults to stepdist_a
      bed_tilt_x        - 0.0 (mm/mm, bed tilt along X)
      bed_tilt_y        - 0.0 (mm/mm, bed tilt along Y)
    """
    section = 'delta_true_calibration'
    if not config.has_section(section):
        raise ValueError(
            "Config file is missing [delta_true_calibration] section.\n"
            "See the script comments for the required format.")
    p = _load_calibration_section(config, section)
    return TrueDeltaCalibration(**p)


def _load_calibration_section(config, section):
    """Load delta calibration parameters from a named config section.

    Helper used by load_true_calibration and load_printer_calibration.
    Returns a dict of parsed parameter values.
    """
    _REQUIRED = object()

    def getfloat(key, fallback=_REQUIRED):
        if config.has_option(section, key):
            return config.getfloat(section, key)
        if fallback is not _REQUIRED:
            return fallback
        raise ValueError(
            "Missing required parameter '%s' in [%s]" % (key, section))

    arm_shared = getfloat('arm_length', fallback=None)
    arm_a = getfloat('arm_length_a', fallback=arm_shared)
    if arm_a is None:
        raise ValueError(
            "Must specify 'arm_length' or 'arm_length_a' in [%s]" % section)
    arm_b = getfloat('arm_length_b', fallback=arm_a)
    arm_c = getfloat('arm_length_c', fallback=arm_a)

    radius = getfloat('delta_radius')

    endstop_a = getfloat('endstop_a')
    endstop_b = getfloat('endstop_b', endstop_a)
    endstop_c = getfloat('endstop_c', endstop_a)

    angle_a = getfloat('angle_a', 210.)
    angle_b = getfloat('angle_b', 330.)
    angle_c = getfloat('angle_c', 90.)

    stepdist_a = getfloat('stepdist_a', 0.0001)
    stepdist_b = getfloat('stepdist_b', stepdist_a)
    stepdist_c = getfloat('stepdist_c', stepdist_a)

    bed_tilt_x = getfloat('bed_tilt_x', 0.)
    bed_tilt_y = getfloat('bed_tilt_y', 0.)

    return dict(
        radius=radius,
        angles=[angle_a, angle_b, angle_c],
        arms=[arm_a, arm_b, arm_c],
        endstops=[endstop_a, endstop_b, endstop_c],
        stepdists=[stepdist_a, stepdist_b, stepdist_c],
        bed_tilt_x=bed_tilt_x,
        bed_tilt_y=bed_tilt_y,
    )


def load_printer_calibration(config):
    """Load initial printer calibration from the [delta_printer_calibration]
    section.

    This represents the starting parameters for the calibration optimization
    (i.e. what the printer "thinks" it is before running DELTA_CALIBRATE).
    Returns a DeltaCalibrationParams, or None if the section is absent.

    Required parameters:
      delta_radius      - radial distance from center to tower positions (mm)
      endstop_a         - position_endstop for tower A (mm)
      arm_length        - arm length shared by all towers, OR per-tower
                          arm_length_a / arm_length_b / arm_length_c (mm)

    Optional parameters (defaults shown):
      endstop_b/c       - defaults to endstop_a
      arm_length_b/c    - defaults to arm_length_a (or arm_length)
      angle_a           - 210.0 (degrees)
      angle_b           - 330.0 (degrees)
      angle_c           - 90.0 (degrees)
      stepdist_a        - 0.0001 (mm/step)
      stepdist_b/c      - defaults to stepdist_a
      bed_tilt_x        - 0.0 (mm/mm, initial bed tilt along X)
      bed_tilt_y        - 0.0 (mm/mm, initial bed tilt along Y)
    """
    section = 'delta_printer_calibration'
    if not config.has_section(section):
        return None
    p = _load_calibration_section(config, section)
    return DeltaCalibrationParams(**p)


######################################################################
# Probe height measurement generation
######################################################################

def _ring_xy(probe_radius, scale, n_points, start_offset_deg=0.):
    """Return XY coordinates for one ring of equally-spaced points."""
    r = probe_radius * scale
    return [
        (math.cos(math.radians(90. + start_offset_deg + 360. * i / n_points)) * r,
         math.sin(math.radians(90. + start_offset_deg + 360. * i / n_points)) * r)
        for i in range(n_points)
    ]


def _ring_specs(pattern):
    """Return [(scale, n_points, start_offset_deg), ...] for ring patterns."""
    if pattern == 'ring1by12':
        return [(1.0, 12, 0.)]
    if pattern == 'ring3by12':
        return [(1.0, 12, 0.), (0.7, 12, 15.), (0.3, 12, 0.)]
    raise ValueError("Unknown ring pattern: %s" % pattern)


def generate_probe_points(probe_radius, pattern='hex39'):
    """Generate probe point XY coordinates for the given pattern.

    pattern - one of 'hex39', 'defaultklipper', 'ring1by12', 'ring3by12'

    hex39:         39-point hexagon pattern from delta_calibrate.py
    defaultklipper: 7-point pattern (center + 6 scattered ring points)
    ring1by12:     13-point pattern (center + 12-point outer ring)
    ring3by12:     37-point pattern (center + three 12-point rings at 1.0,
                   0.7, and 0.3 of probe_radius)
    """
    if pattern == 'hex39':
        return [(x * probe_radius, y * probe_radius)
                for x, y in HexagonProbePattern_39points]
    if pattern == 'defaultklipper':
        scatter = [.95, .90, .85, .70, .75, .80]
        points = [(0., 0.)]
        for i, s in enumerate(scatter):
            r = math.radians(90. + 60. * i)
            points.append((math.cos(r) * probe_radius * s,
                           math.sin(r) * probe_radius * s))
        return points
    # Ring patterns: center + one or more equally-spaced rings
    points = [(0., 0.)]
    for scale, n, offset in _ring_specs(pattern):
        points.extend(_ring_xy(probe_radius, scale, n, offset))
    return points


def generate_height_measurements(true_cal, probe_points, noise=0.):
    """Generate simulated probe height measurements at each probe point.

    For each (x, y) probe position the bed surface height is determined by
    the bed tilt parameters, and the corresponding stepper stable position is
    computed using the true calibration geometry.

    Returns a list of (z_offset, stable_position) tuples, where:
      z_offset        - always 0.0 (the probe triggers at the bed surface)
      stable_position - 3-tuple of stepper steps from endstop (tower a, b, c)

    On a physical printer the probe always fires at the bed surface, so the
    recorded height offset is always 0.  Any bed tilt is captured entirely in
    the stable stepper positions, not in the z_offset value.

    When noise > 0 a uniform random perturbation in [-noise, +noise] mm is
    added to the probe trigger height, simulating probe repeatability error.
    The perturbation shifts all three stepper stable positions by the same
    amount (it is a pure Z-direction noise), leaving z_offset at 0.
    """
    measurements = []
    for x, y in probe_points:
        z = true_cal.bed_z(x, y)
        if noise:
            z += random.uniform(-noise, noise)
        stable_pos = true_cal.calc_stable_position([x, y, z])
        measurements.append((0., stable_pos))
    return measurements


######################################################################
# Distance measurement generation
######################################################################

def generate_distance_measurements(true_cal, initial_cal, scale=1.0, noise=0.):
    """Generate simulated distance measurements from calibration object ridges.

    Simulates the physical DELTA_ANALYZE measurement process:
      1. The printer uses *initial_cal* to compute the stepper stable positions
         for each nominal ridge coordinate (what the printer would command).
      2. *true_cal* resolves those stepper positions back to actual physical
         Cartesian coordinates (where the toolhead truly ends up).
      3. The caliper distance is the XY distance between those true positions.

    This reflects the real procedure where the printer moves to nominal ridge
    positions using its current (possibly imperfect) calibration, and the
    operator measures the actual physical distances with a caliper.

    Returns a list of (distance, stable_pos1, stable_pos2) tuples in the
    same format that DELTA_ANALYZE writes to the [delta_calibrate] section.
    The list contains 12 entries (6 center + 6 outer), matching the output
    of measurements_to_distances() in delta_calibrate.py.

    When noise > 0 a uniform random perturbation in [-noise, +noise] mm is
    added to each recorded distance, simulating caliper measurement error.
    The stable positions are not affected; only the measured distance value
    is perturbed.
    """
    obj_angles = list(map(math.radians, MEASURE_ANGLES))
    xy_angles = list(zip(map(math.cos, obj_angles), map(math.sin, obj_angles)))

    inner_ridge = MEASURE_RIDGE_RADIUS * scale
    outer_ridge = (MEASURE_OUTER_RADIUS + MEASURE_RIDGE_RADIUS) * scale
    outer_center = MEASURE_OUTER_RADIUS * scale

    def ridge_stable_and_pos(xy):
        """Return (stable_pos, actual_cartesian) for a ridge at XY position.

        The stable position is computed via *initial_cal* (what the printer
        commands); the actual Cartesian position is resolved via *true_cal*
        (where the toolhead physically ends up).
        """
        x, y = xy
        z = true_cal.bed_z(x, y)
        spos = initial_cal.calc_stable_position([x, y, z])
        actual_pos = true_cal.get_position_from_stable(spos)
        return spos, actual_pos

    # --- Center measurements: one per spoke (inner to outer) ---
    inner_pos = [(ax * inner_ridge, ay * inner_ridge)
                 for ax, ay in xy_angles]
    outer_pos = [(ax * outer_ridge, ay * outer_ridge)
                 for ax, ay in xy_angles]

    center_distances = []
    for ip, op in zip(inner_pos, outer_pos):
        spos1, apos1 = ridge_stable_and_pos(ip)
        spos2, apos2 = ridge_stable_and_pos(op)
        # Caliper (horizontal) distance between the true physical positions
        dist = math.sqrt((apos2[0] - apos1[0])**2 + (apos2[1] - apos1[1])**2)
        if noise:
            dist += random.uniform(-noise, noise)
        center_distances.append((dist, spos1, spos2))

    # --- Outer measurements: pairs of adjacent spokes ---
    # Angles are shifted by 2 positions so each pair straddles a tower sector
    shifted_angles = xy_angles[2:] + xy_angles[:2]
    start_pos = [(ax * outer_center, ay * outer_center)
                 for ax, ay in xy_angles]
    first_pos = [(ax * inner_ridge + spx, ay * inner_ridge + spy)
                 for (ax, ay), (spx, spy) in zip(shifted_angles, start_pos)]
    second_pos = [(ax * outer_ridge + spx, ay * outer_ridge + spy)
                  for (ax, ay), (spx, spy) in zip(shifted_angles, start_pos)]

    outer_distances = []
    for fp, sp_pt in zip(first_pos, second_pos):
        spos1, apos1 = ridge_stable_and_pos(fp)
        spos2, apos2 = ridge_stable_and_pos(sp_pt)
        # Caliper (horizontal) distance between the true physical positions
        dist = math.sqrt((apos2[0] - apos1[0])**2 + (apos2[1] - apos1[1])**2)
        if noise:
            dist += random.uniform(-noise, noise)
        outer_distances.append((dist, spos1, spos2))

    return center_distances + outer_distances


def generate_ring_distance_measurements(true_cal, initial_cal, probe_radius,
                                        rings, noise=0.):
    """Generate distance measurements for ring-based probe patterns.

    For each ring defined by (scale, n_points, start_offset_deg), generates:
      - n_points distances from origin to each ring point
      - n_points distances between adjacent ring points (wrapping)

    Uses the prototypic two-step approach: initial_cal determines the stepper
    stable positions for each nominal coordinate; true_cal resolves those
    positions back to actual physical Cartesian coordinates; distances are
    the horizontal (XY) separations between the true positions.
    """
    def ring_stable_and_pos(xy):
        x, y = xy
        z = true_cal.bed_z(x, y)
        spos = initial_cal.calc_stable_position([x, y, z])
        actual = true_cal.get_position_from_stable(spos)
        return spos, actual

    origin_spos, origin_apos = ring_stable_and_pos((0., 0.))
    distances = []
    for scale, n_points, offset in rings:
        ring_pts = _ring_xy(probe_radius, scale, n_points, offset)
        ring_data = [ring_stable_and_pos(pt) for pt in ring_pts]
        # Origin to each ring point
        for spos, apos in ring_data:
            dist = math.sqrt((apos[0] - origin_apos[0]) ** 2
                             + (apos[1] - origin_apos[1]) ** 2)
            if noise:
                dist += random.uniform(-noise, noise)
            distances.append((dist, origin_spos, spos))
        # Adjacent ring point pairs (wrapping)
        for j in range(n_points):
            spos1, apos1 = ring_data[j]
            spos2, apos2 = ring_data[(j + 1) % n_points]
            dist = math.sqrt((apos2[0] - apos1[0]) ** 2
                             + (apos2[1] - apos1[1]) ** 2)
            if noise:
                dist += random.uniform(-noise, noise)
            distances.append((dist, spos1, spos2))
    return distances


######################################################################
# Calibration optimization (mimicking DELTA_CALIBRATE / DELTA_ANALYZE)
######################################################################

# How much to weight distance measurements relative to height measurements.
# Matches the MEASURE_WEIGHT constant in delta_calibrate.py.


MEASURE_WEIGHT = 0.5


def coordinate_descent(adj_params, params, error_func, initial_dp=None):
    """Perform coordinate descent to minimize error_func.

    Mirrors the coordinate_descent() function in klippy/mathutil.py but runs
    in the current process without any klipper runtime dependencies.
    initial_dp allows per-parameter initial step sizes (defaults to 1.0).
    Returns (best_params, best_error, rounds).
    """
    params = dict(params)
    dp = {p: (initial_dp.get(p, 1.) if initial_dp else 1.)
          for p in adj_params}
    best_err = error_func(params)
    threshold = 0.00001
    rounds = 0
    while sum(dp.values()) > threshold and rounds < 50000:
        rounds += 1
        for param_name in adj_params:
            orig = params[param_name]
            params[param_name] = orig + dp[param_name]
            err = error_func(params)
            if err < best_err:
                best_err = err
                dp[param_name] *= 1.1
                continue
            params[param_name] = orig - dp[param_name]
            err = error_func(params)
            if err < best_err:
                best_err = err
                dp[param_name] *= 1.1
                continue
            params[param_name] = orig
            dp[param_name] *= 0.9
        if rounds % 100 == 0:
            print("  Round %d: error=%.15f" %
                  (rounds, best_err), end='\r', flush=True)
    return params, best_err, rounds


def calibrate_delta(printer_cal, height_positions, distances):
    """Run delta calibration optimization, mimicking DELTA_CALIBRATE/DELTA_ANALYZE.

    printer_cal:      DeltaCalibrationParams - initial (starting) calibration
    height_positions: list of (z_offset, stable_pos) from probe measurements
    distances:        list of (dist, stable_pos1, stable_pos2) from distance
                      measurements (empty list for basic DELTA_CALIBRATE only)

    When distances are provided (is_extended=True), arm lengths and angles are
    also adjusted (DELTA_ANALYZE behavior), mirroring the extended calibration
    in delta.py's coordinate_descent_params(is_extended=True).

    bed_tilt_x and bed_tilt_y are always included as adjustment parameters so
    bed tilt is separated from the delta geometry during calibration.

    Returns (new_cal, best_error, rounds) where new_cal is a
    DeltaCalibrationParams with the optimized parameters.
    """
    is_extended = bool(distances)
    orig_cal = printer_cal
    adj_params, params = orig_cal.coordinate_descent_params(is_extended)

    # Use appropriate initial step sizes for each parameter type.
    # bed_tilt parameters have much smaller typical values (~0.001 mm/mm)
    # than geometric parameters (~1 mm / ~1 deg), so start with a smaller
    # initial step to avoid wasting rounds shrinking dp down from 1.0.
    initial_dp = {p: 1. for p in adj_params}
    for p in adj_params:
        if 'bed_tilt' in p:
            initial_dp[p] = 0.01

    z_weight = 1.
    if distances:
        z_weight = len(distances) / (MEASURE_WEIGHT * len(height_positions))

    def delta_errorfunc(params):
        try:
            cal = orig_cal.new_calibration(params)
            getpos = cal.get_position_from_stable
            bed_tilt_x = params['bed_tilt_x']
            bed_tilt_y = params['bed_tilt_y']
            # Compute reconstructed cartesian positions for all probe points
            positions = [getpos(spos) for _, spos in height_positions]
            # Find the best-fit tilted plane z = bed_tilt_x*x + bed_tilt_y*y + tz
            # and compute sum of squared deviations.  The optimal tz is the mean
            # of the residuals (eliminates the constant offset analytically).
            residuals = [z - bed_tilt_x * x - bed_tilt_y * y
                         for x, y, z in positions]
            z_mean = sum(residuals) / len(residuals)
            # height_error = sum((r - z_mean) ** 2 for r in residuals)
            height_error = sum((r) ** 2 for r in residuals)
            
            total_error = height_error * z_weight
            
            #orgin should be lowest travel point in xy plane
            towers = cal.calc_tower_from_position((0., 0., 0.))
            
            TOWER_OFFSET = .10 #mm offset to add to each tower height for distance error calculation 
            #add offset to each tower height
            t_a = [towers[0] + TOWER_OFFSET, towers[1], towers[2]]            
            t_b = [towers[0], towers[1] + TOWER_OFFSET, towers[2]]
            t_c = [towers[0], towers[1], towers[2] + TOWER_OFFSET]
            
            z_a = cal.get_position_from_tower(t_a)[2]
            z_b = cal.get_position_from_tower(t_b)[2]
            z_c = cal.get_position_from_tower(t_c)[2]
            
            total_error += (z_a - z_b) ** 2 + (z_b - z_c) ** 2 + (z_c - z_a) ** 2
                                    
            # Distance error: horizontal (XY) distance between each ridge pair
            for dist, sp1, sp2 in distances:
                x1, y1, _z1 = getpos(sp1)
                x2, y2, _z2 = getpos(sp2)
                d = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
                total_error += (d - dist) ** 2
            return total_error
        except (ValueError, ZeroDivisionError):
            return 9e18

    new_params, best_err, rounds = coordinate_descent(
        adj_params, params, delta_errorfunc, initial_dp=initial_dp)
    new_cal = orig_cal.new_calibration(new_params)
    return new_cal, best_err, rounds


def _print_calibration(cal):
    """Print calibration parameters in a human-readable format."""
    print("  delta_radius : %.6f mm" % cal.radius)
    print("  arm lengths  : %.6f, %.6f, %.6f mm" % tuple(cal.arms))
    print("  angles       : %.6f, %.6f, %.6f deg" % tuple(cal.angles))
    print("  endstops     : %.6f, %.6f, %.6f mm" % tuple(cal.endstops))
    print("  bed_tilt_x   : %.6f mm/mm" % cal.bed_tilt_x)
    print("  bed_tilt_y   : %.6f mm/mm" % cal.bed_tilt_y)


def simulate_bed_mesh(true_cal, cal_params, probe_radius, count=7):
    """Simulate a bed mesh using the given calibration parameters.

    Generates a rectangular grid of (count x count) points clipped to the
    circular probe area of the given radius.  For each point the probe is
    assumed to fire at the true bed surface; the calibrated printer then
    reconstructs a cartesian Z from the resulting stable stepper positions.

    Returns a list of (x, y, mesh_z, true_z) tuples where:
      mesh_z - Z value the calibrated printer would record at (x, y)
      true_z - true bed height from true_cal.bed_z(x, y)
    """
    results = []
    for i in range(count):
        for j in range(count):
            x = probe_radius * (-1. + 2. * i / (count - 1))
            y = probe_radius * (-1. + 2. * j / (count - 1))
            if x ** 2 + y ** 2 > probe_radius ** 2:
                continue
            true_z = true_cal.bed_z(x, y)
            stable_pos = true_cal.calc_stable_position([x, y, true_z])
            _mx, _my, mesh_z = cal_params.get_position_from_stable(stable_pos)
            results.append((x, y, mesh_z, true_z))
    return results


def _print_mesh_metrics(label, mesh_results):
    """Print bed mesh min/max statistics for a simulate_bed_mesh() result."""
    mesh_z = [mz for _x, _y, mz, _tz in mesh_results]
    errors = [mz - tz for _x, _y, mz, tz in mesh_results]
    print("%s (%d points):" % (label, len(mesh_results)))
    print("  Mesh Z   : min %+.6f  max %+.6f  range %.6f mm"
          % (min(mesh_z), max(mesh_z), max(mesh_z) - min(mesh_z)))
    print("  True bed error: min %+.6f  max %+.6f  range %.6f mm"
          % (min(errors), max(errors), max(errors) - min(errors)))


def _print_distance_metrics(label, cal, distances):
    """Print distance measurement accuracy metrics for a calibration.

    For each (measured_dist, spos1, spos2), reconstructs the Cartesian
    positions from the stepper stable positions using *cal*, computes the
    horizontal distance between them, and reports the signed error
    (reconstructed - measured).
    """
    errors = []
    for dist, sp1, sp2 in distances:
        x1, y1, _z1 = cal.get_position_from_stable(sp1)
        x2, y2, _z2 = cal.get_position_from_stable(sp2)
        d = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        errors.append(d - dist)
    rms = math.sqrt(sum(e ** 2 for e in errors) / len(errors))
    print("%s (%d measurements):" % (label, len(distances)))
    print("  Distance error: min %+.6f  max %+.6f  rms %.6f mm"
          % (min(errors), max(errors), rms))


######################################################################
# Carriage-sum chart generation
######################################################################

def _carriage_sum_grid(cal, half_range=1., steps=200):
    """Compute a 2-D grid of the sum of the three tower carriage heights.

    For each (x, y) on a uniform grid spanning [-half_range, +half_range],
    the toolhead is placed at z=0 and the three carriage Z-axis positions
    (mm above the zero datum on each tower axis) are summed.

    Returns (xs, ys, zs) where xs/ys are 1-D coordinate arrays and zs is
    a 2-D array of shape (len(ys), len(xs)) suitable for imshow/pcolormesh.
    """
    import numpy as np
    xs = np.linspace(-half_range, half_range, steps)
    ys = np.linspace(-half_range, half_range, steps)
    zs = np.full((steps, steps), float('nan'))
    # Precompute per-tower constants
    towers = cal.towers
    arms = cal.arms
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            try:
                # Carriage Z for tower k when toolhead is at (x, y, 0)
                total = sum(
                    math.sqrt(a**2 - (tx - x)**2 - (ty - y)**2)
                    for (tx, ty), a in zip(towers, arms)
                )
                zs[j, i] = total
            except ValueError:
                pass  # outside reachable envelope
    return xs, ys, zs


def generate_carriage_charts(cal_initial, cal_final, out_prefix='carriage_sum',
                              half_range=1., steps=200):
    """Generate and save two PNG charts of the carriage-height sum.

    Each chart covers ±half_range mm in X and Y at z=0.  The colour map shows
    the sum of the three tower carriage Z positions (mm above datum) at each
    (x, y) point, with contour lines overlaid.

    Saves <out_prefix>_initial.png and <out_prefix>_final.png.
    """
    if not HAS_MATPLOTLIB:
        print("Warning: matplotlib not available -- skipping carriage charts.")
        return
    import numpy as np

    datasets = [
        (cal_initial, 'Initial calibration', out_prefix + '_initial.png'),
        (cal_final,   'Final calibration',   out_prefix + '_final.png'),
    ]
    for cal, title, filename in datasets:
        xs, ys, zs = _carriage_sum_grid(cal, half_range=half_range,
                                         steps=steps)
        fig, ax = plt.subplots(figsize=(7, 6))
        pcm = ax.pcolormesh(xs, ys, zs, shading='auto', cmap='viridis')
        # Contour lines on top
        valid = ~np.isnan(zs)
        if valid.any():
            levels = np.linspace(np.nanmin(zs), np.nanmax(zs), 15)
            ax.contour(xs, ys, zs, levels=levels, colors='white',
                       linewidths=0.5, alpha=0.6)
        cb = fig.colorbar(pcm, ax=ax)
        cb.set_label('Sum of carriage heights (mm)')
        ax.set_title(title)
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_aspect('equal')
        ax.set_xlim(-half_range, half_range)
        ax.set_ylim(-half_range, half_range)
        fig.tight_layout()
        fig.savefig(filename, dpi=150)
        plt.close(fig)
        print("Wrote carriage-sum chart: %s" % filename)


######################################################################
# Output formatting and config file update
######################################################################

def _fmt_stable_pos(spos):
    """Format a stable position 3-tuple in the klipper config format."""
    return "%.3f,%.3f,%.3f" % tuple(spos)


def _fmt_float(value):
    """Format a float, trimming unnecessary trailing zeros.

    Normalizes negative zero to zero to avoid the display of '-0'.
    """
    if value == 0.:
        value = 0.
    s = "%.6g" % value
    return s


def build_delta_calibrate_lines(height_measurements, distance_measurements):
    """Return a list of 'key = value' strings for the [delta_calibrate] section.

    The output format matches what klipper's DeltaCalibrate.save_state()
    writes so the generated data can be loaded by klipper directly.
    """
    lines = ["[delta_calibrate]"]
    for i, (z_offset, spos) in enumerate(height_measurements):
        lines.append("height%d = %s" % (i, _fmt_float(z_offset)))
        lines.append("height%d_pos = %s" % (i, _fmt_stable_pos(spos)))
    for i, (dist, spos1, spos2) in enumerate(distance_measurements):
        lines.append("distance%d = %s" % (i, _fmt_float(dist)))
        lines.append("distance%d_pos1 = %s" % (i, _fmt_stable_pos(spos1)))
        lines.append("distance%d_pos2 = %s" % (i, _fmt_stable_pos(spos2)))
    return lines


def _replace_save_config_section(save_config_lines, section_name, new_lines):
    """Replace or append a section in the SAVE_CONFIG line list.

    save_config_lines - list of lines (without the #*# prefix) from the block
    section_name      - section name without brackets, e.g. 'delta_calibrate'
    new_lines         - list of replacement lines (includes '[section]' header)

    Returns a new list of lines with the section replaced.
    """
    header = "[%s]" % section_name
    result = []
    i = 0
    replaced = False
    # Copy lines up to (not including) the target section
    while i < len(save_config_lines):
        line = save_config_lines[i]
        if line.strip() == header:
            # Skip the old section lines
            replaced = True
            # Insert new section
            result.extend(l + "\n" for l in new_lines)
            i += 1
            # Skip existing lines until next section or end
            while i < len(save_config_lines):
                l = save_config_lines[i]
                if l.startswith('[') and l.strip().endswith(']'):
                    break
                i += 1
        else:
            result.append(line)
            i += 1
    if not replaced:
        # Append the new section at the end
        if result and result[-1].strip():
            result.append("\n")
        result.extend(l + "\n" for l in new_lines)
    return result


def update_config_file(filename, orig_lines, save_config_start,
                       height_measurements, distance_measurements):
    """Update the printer.cfg file with the generated calibration data.

    Replaces (or appends) the [delta_calibrate] section inside the
    SAVE_CONFIG block and writes the updated file back to disk.
    """
    new_delta_lines = build_delta_calibrate_lines(
        height_measurements, distance_measurements)

    # Lines before the SAVE_CONFIG block (keep unchanged)
    if save_config_start is not None:
        pre_lines = orig_lines[:save_config_start]
        save_block_lines = orig_lines[save_config_start:]
    else:
        pre_lines = list(orig_lines)
        save_block_lines = []

    # Extract existing SAVE_CONFIG content (stripping #*# prefix)
    existing_save_lines = _parse_save_config_block(save_block_lines)

    # Replace / append [delta_calibrate] in the SAVE_CONFIG content
    updated_save_lines = _replace_save_config_section(
        existing_save_lines, 'delta_calibrate', new_delta_lines)

    # Write the file back
    with open(filename, 'w') as f:
        f.writelines(pre_lines)
        f.write(SAVE_CONFIG_HEADER + "\n")
        f.write(SAVE_CONFIG_WARNING + "\n")
        f.write("#*#\n")
        for line in updated_save_lines:
            stripped = line.rstrip('\n')
            if stripped:
                f.write("#*# " + stripped + "\n")
            else:
                f.write("#*#\n")


def write_output_file(filename, height_measurements, distance_measurements):
    """Write only the [delta_calibrate] section data to a standalone file."""
    lines = build_delta_calibrate_lines(height_measurements,
                                        distance_measurements)
    with open(filename, 'w') as f:
        for line in lines:
            f.write(line + "\n")


######################################################################
# Main entry point
######################################################################

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate simulated delta calibration probe height data and\n"
            "distance measurements from a [delta_true_calibration] config\n"
            "section, save them to the [delta_calibrate] section of the\n"
            "printer.cfg SAVE_CONFIG block, and optionally run the calibration\n"
            "optimization to mimic DELTA_CALIBRATE / DELTA_ANALYZE.\n"
            "\n"
            "The [delta_true_calibration] section defines the physical\n"
            "ground-truth parameters of the delta printer:\n"
            "  delta_radius, arm_length, endstop_a/b/c,\n"
            "  angle_a/b/c, stepdist_a/b/c, bed_tilt_x, bed_tilt_y\n"
            "\n"
            "The optional [delta_printer_calibration] section defines the\n"
            "initial (starting) calibration for the optimization:\n"
            "  same parameters as [delta_true_calibration]\n"
            "If present, the script runs coordinate descent to find the best\n"
            "calibration parameters and prints the results."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('config', help='Printer config file (printer.cfg)')
    parser.add_argument(
        '-o', '--output', default=None,
        help='Write [delta_calibrate] data to this file instead of updating '
             'the input config in place')
    parser.add_argument(
        '--pattern',
        choices=['hex39', 'defaultklipper', 'ring1by12', 'ring3by12'],
        default='hex39',
        help='Probe point pattern to simulate (default: hex39).  '
             'hex39/defaultklipper use calibration-object ridge distances; '
             'ring1by12/ring3by12 use origin-to-ring and adjacent-ring '
             'point distances.')
    parser.add_argument(
        '--no-distances', action='store_true',
        help='Skip generating distance measurements (only height data)')
    parser.add_argument(
        '--scale', type=float, default=1.0,
        help='Calibration object scale factor (default: 1.0)')
    parser.add_argument(
        '--no-calibrate', action='store_true',
        help='Skip the calibration optimization step even if '
             '[delta_printer_calibration] is present')
    parser.add_argument(
        '--noise', type=float, default=0.0, metavar='MM',
        help='Maximum magnitude of uniform random noise added to simulated '
             'measurements (default: 0.0).  Probe height measurements are '
             'perturbed by a random value in [-MM, +MM] mm applied to the '
             'probe trigger height; distance measurements are perturbed by a '
             'random value in [-MM, +MM] mm applied to the recorded distance.')
    parser.add_argument(
        '--verbose', action='store_true',
        help='Print generated measurements and calibration results in detail')
    opts = parser.parse_args()

    if not os.path.exists(opts.config):
        sys.stderr.write("Error: config file not found: %s\n" % opts.config)
        sys.exit(1)

    # Load and parse configuration
    try:
        config, orig_lines, save_config_start = parse_klipper_config(
            opts.config)
        true_cal = load_true_calibration(config)
        printer_cal = load_printer_calibration(config)
    except (ValueError, configparser.Error) as e:
        sys.stderr.write("Error reading config: %s\n" % e)
        sys.exit(1)

    # Get probe radius from [delta_calibrate] section
    probe_radius = 50.
    if config.has_section('delta_calibrate'):
        probe_radius = config.getfloat('delta_calibrate', 'radius',
                                       fallback=50.)

    # Print summary of true (ground-truth) parameters
    print("True calibration parameters:")
    print("  delta_radius : %.4f mm" % true_cal.radius)
    print("  arm lengths  : %.4f, %.4f, %.4f mm" % tuple(true_cal.arms))
    print("  angles       : %.4f, %.4f, %.4f deg" % tuple(true_cal.angles))
    print("  endstops     : %.4f, %.4f, %.4f mm" % tuple(true_cal.endstops))
    print("  step dists   : %.6f, %.6f, %.6f mm/step"
          % tuple(true_cal.stepdists))
    print("  bed_tilt_x   : %.6f mm/mm" % true_cal.bed_tilt_x)
    print("  bed_tilt_y   : %.6f mm/mm" % true_cal.bed_tilt_y)
    print("Probe radius   : %.2f mm" % probe_radius)
    print("Probe pattern  : %s" % opts.pattern)
    if opts.scale != 1.0:
        print("Object scale   : %.4f" % opts.scale)
    if opts.noise:
        print("Noise magnitude: %.6f mm" % opts.noise)

    # Generate probe height measurements
    probe_points = generate_probe_points(probe_radius, opts.pattern)
    height_measurements = generate_height_measurements(
        true_cal, probe_points, opts.noise)

    # Generate distance measurements (unless disabled).
    # Use printer_cal as the initial calibration so stable positions reflect
    # what the printer would command; fall back to true_cal when no printer
    # calibration section is present (equivalent to a perfect printer).
    distance_measurements = []
    if not opts.no_distances:
        initial_cal = printer_cal if printer_cal is not None else true_cal
        if opts.pattern in ('hex39', 'defaultklipper'):
            distance_measurements = generate_distance_measurements(
                true_cal, initial_cal, opts.scale, opts.noise)
        else:
            rings = _ring_specs(opts.pattern)
            distance_measurements = generate_ring_distance_measurements(
                true_cal, initial_cal, probe_radius, rings, opts.noise)

    print("\nGenerated %d height measurements and %d distance measurements"
          % (len(height_measurements), len(distance_measurements)))

    if opts.verbose:
        # Display generated data
        print("\nHeight measurements:")
        for i, (z, spos) in enumerate(height_measurements):
            print("  height%d = %s  pos = %s"
                % (i, _fmt_float(z), _fmt_stable_pos(spos)))
        if distance_measurements:
            print("Distance measurements:")
            for i, (d, s1, s2) in enumerate(distance_measurements):
                print("  distance%d = %s  pos1 = %s  pos2 = %s"
                    % (i, _fmt_float(d), _fmt_stable_pos(s1),
                        _fmt_stable_pos(s2)))

    # Write output
    if opts.output:
        write_output_file(opts.output, height_measurements,
                          distance_measurements)
        print("\nWrote [delta_calibrate] data to: %s" % opts.output)
    else:
        update_config_file(opts.config, orig_lines, save_config_start,
                           height_measurements, distance_measurements)
        print("\nUpdated SAVE_CONFIG block in: %s" % opts.config)

    # Run calibration optimization if [delta_printer_calibration] is present
    if printer_cal is not None and not opts.no_calibrate:
        print("\n--- Calibration optimization ---")
        print("Initial printer calibration (starting point):")
        _print_calibration(printer_cal)

        calibration_type = ("extended (DELTA_CALIBRATE + DELTA_ANALYZE)"
                            if distance_measurements
                            else "basic (DELTA_CALIBRATE)")
        print("\nRunning %s calibration..." % calibration_type)

        new_cal, best_err, rounds = calibrate_delta(
            printer_cal, height_measurements, distance_measurements)

        print("Completed in %d rounds, final error: %.6g" % (rounds, best_err))
        print("\nCalibrated parameters:")
        _print_calibration(new_cal)

        print("\nDelta from true parameters:")
        print("  delta_radius : %+.6f mm" % (new_cal.radius - true_cal.radius))
        for i, axis in enumerate('abc'):
            print("  arm_%s        : %+.6f mm"
                  % (axis, new_cal.arms[i] - true_cal.arms[i]))
        for i, axis in enumerate('abc'):
            print("  angle_%s      : %+.6f deg"
                  % (axis, new_cal.angles[i] - true_cal.angles[i]))
        for i, axis in enumerate('abc'):
            print("  endstop_%s   : %+.6f mm"
                  % (axis, new_cal.endstops[i] - true_cal.endstops[i]))
        print("  bed_tilt_x   : %+.6f mm/mm"
              % (new_cal.bed_tilt_x - true_cal.bed_tilt_x))
        print("  bed_tilt_y   : %+.6f mm/mm"
              % (new_cal.bed_tilt_y - true_cal.bed_tilt_y))

        print("\nSuggested printer config changes:")
        print("[printer]")
        print("  delta_radius: %.6f" % new_cal.radius)
        for i, axis in enumerate('abc'):
            print("[stepper_%s]" % axis)
            print("  angle: %.6f" % new_cal.angles[i])
            print("  arm_length: %.6f" % new_cal.arms[i])
            print("  position_endstop: %.6f" % new_cal.endstops[i])
        print("[bed_tilt]")
        print("  x_adjust: %.6f" % new_cal.bed_tilt_x)
        print("  y_adjust: %.6f" % new_cal.bed_tilt_y)

        # Distance measurement accuracy: compare initial and calibrated results
        if distance_measurements:
            print("\n--- Distance measurement accuracy ---")
            _print_distance_metrics("Before calibration", printer_cal,
                                    distance_measurements)
            _print_distance_metrics("After calibration", new_cal,
                                    distance_measurements)

        # Bed mesh simulation: compare initial and calibrated results
        print("\n--- Bed mesh simulation ---")
        initial_mesh = simulate_bed_mesh(true_cal, printer_cal, probe_radius)
        _print_mesh_metrics("Before calibration", initial_mesh)
        final_mesh = simulate_bed_mesh(true_cal, new_cal, probe_radius)
        _print_mesh_metrics("After calibration", final_mesh)

        # Carriage-sum charts
        print("\n--- Carriage-sum charts ---")
        chart_prefix = os.path.splitext(opts.config)[0] + '_carriage_sum'
        generate_carriage_charts(printer_cal, new_cal,
                                 out_prefix=chart_prefix)


if __name__ == '__main__':
    main()
