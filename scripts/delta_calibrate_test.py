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

# Calibration object geometry constants (matching delta_calibrate.py)
MEASURE_ANGLES = [210., 270., 330., 30., 90., 150.]
MEASURE_OUTER_RADIUS = 65.
MEASURE_RIDGE_RADIUS = 5. - .5  # 4.5 mm

# Probe pattern copied verbatim from delta_calibrate.py (named "37points" there
# but actually contains 39 coordinate pairs).  Normalized coordinates are
# multiplied by probe_radius to get the actual probe positions.
HexagonProbePattern_37points = [
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
        self.angles = angles
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

    def coordinate_descent_params(self, is_extended):
        """Return (adj_params, params) for coordinate descent optimization.

        adj_params: tuple of parameter names to adjust during descent
        params:     dict of current parameter values

        bed_tilt_x and bed_tilt_y are always included as adjustment parameters.
        With is_extended=True (distance measurements available), arm lengths
        and tower angles are also included, matching DELTA_ANALYZE behavior.
        """
        adj_params = ('radius',
                      'endstop_a', 'endstop_b', 'endstop_c',
                      'bed_tilt_x', 'bed_tilt_y')
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

def generate_probe_points(probe_radius):
    """Generate probe point XY coordinates matching DELTA_CALIBRATE's pattern.

    Uses the 37-point hexagon pattern from delta_calibrate.py, scaled by the
    given probe_radius.
    """
    return [(x * probe_radius, y * probe_radius)
            for x, y in HexagonProbePattern_37points]


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

def generate_distance_measurements(true_cal, scale=1.0, noise=0.):
    """Generate simulated distance measurements from calibration object ridges.

    Computes stable positions for the inner and outer ridge locations of the
    calibration object (docs/prints/calibrate_size.stl) using the true
    calibration, and the physical (caliper) distances between ridge pairs.

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

    # --- Center measurements: one per spoke (inner to outer) ---
    inner_pos = [(ax * inner_ridge, ay * inner_ridge)
                 for ax, ay in xy_angles]
    outer_pos = [(ax * outer_ridge, ay * outer_ridge)
                 for ax, ay in xy_angles]

    center_distances = []
    for ip, op in zip(inner_pos, outer_pos):
        iz = true_cal.bed_z(ip[0], ip[1])
        oz = true_cal.bed_z(op[0], op[1])
        spos1 = true_cal.calc_stable_position([ip[0], ip[1], iz])
        spos2 = true_cal.calc_stable_position([op[0], op[1], oz])
        # Caliper (horizontal) distance between ridge centers along the spoke
        dist = MEASURE_OUTER_RADIUS * scale
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
        fz = true_cal.bed_z(fp[0], fp[1])
        sz = true_cal.bed_z(sp_pt[0], sp_pt[1])
        spos1 = true_cal.calc_stable_position([fp[0], fp[1], fz])
        spos2 = true_cal.calc_stable_position([sp_pt[0], sp_pt[1], sz])
        # Caliper (horizontal) distance between the two outer ridge centers
        dist = math.sqrt((sp_pt[0] - fp[0])**2 + (sp_pt[1] - fp[1])**2)
        if noise:
            dist += random.uniform(-noise, noise)
        outer_distances.append((dist, spos1, spos2))

    return center_distances + outer_distances


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
    while sum(dp.values()) > threshold and rounds < 100:
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
            initial_dp[p] = 0.001

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
            height_error = sum((r - z_mean) ** 2 for r in residuals)
            total_error = height_error * z_weight
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
    if opts.scale != 1.0:
        print("Object scale   : %.4f" % opts.scale)
    if opts.noise:
        print("Noise magnitude: %.6f mm" % opts.noise)

    # Generate probe height measurements
    probe_points = generate_probe_points(probe_radius)
    height_measurements = generate_height_measurements(
        true_cal, probe_points, opts.noise)

    # Generate distance measurements (unless disabled)
    distance_measurements = []
    if not opts.no_distances:
        distance_measurements = generate_distance_measurements(
            true_cal, opts.scale, opts.noise)

    print("\nGenerated %d height measurements and %d distance measurements"
          % (len(height_measurements), len(distance_measurements)))

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
        print("  [printer]")
        print("  delta_radius: %.6f" % new_cal.radius)
        for i, axis in enumerate('abc'):
            print("  [stepper_%s]" % axis)
            print("  angle: %.6f" % new_cal.angles[i])
            print("  arm_length: %.6f" % new_cal.arms[i])
            print("  position_endstop: %.6f" % new_cal.endstops[i])
        print("  [bed_tilt]")
        print("  x_adjust: %.6f" % new_cal.bed_tilt_x)
        print("  y_adjust: %.6f" % new_cal.bed_tilt_y)


if __name__ == '__main__':
    main()
