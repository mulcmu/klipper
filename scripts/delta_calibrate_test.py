#!/usr/bin/env python3
# Delta calibration performance testing tool - Phase 1: Data Generation
#
# Generates simulated probe height measurements and distance measurement
# stable positions from a [delta_true_calibration] config section,
# and saves them to the [delta_calibrate] SAVE_CONFIG block in printer.cfg.
#
# Usage: delta_calibrate_test.py [options] <config_file>
#
# Copyright (C) 2024  Kevin O'Connor <kevin@koconnor.net>
#
# This file may be distributed under the terms of the GNU GPLv3 license.
from __future__ import print_function
import argparse, configparser, io, math, os, sys

# Calibration object geometry constants (matching delta_calibrate.py)
MEASURE_ANGLES = [210., 270., 330., 30., 90., 150.]
MEASURE_OUTER_RADIUS = 65.
MEASURE_RIDGE_RADIUS = 5. - .5  # 4.5 mm

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

    _REQUIRED = object()

    def getfloat(key, fallback=_REQUIRED):
        if config.has_option(section, key):
            return config.getfloat(section, key)
        if fallback is not _REQUIRED:
            return fallback
        raise ValueError(
            "Missing required parameter '%s' in [%s]" % (key, section))

    # Arm lengths: support either a shared arm_length or per-tower values
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

    return TrueDeltaCalibration(
        radius=radius,
        angles=[angle_a, angle_b, angle_c],
        arms=[arm_a, arm_b, arm_c],
        endstops=[endstop_a, endstop_b, endstop_c],
        stepdists=[stepdist_a, stepdist_b, stepdist_c],
        bed_tilt_x=bed_tilt_x,
        bed_tilt_y=bed_tilt_y,
    )


######################################################################
# Probe height measurement generation
######################################################################

def generate_probe_points(probe_radius):
    """Generate the default probe point XY coordinates used by DELTA_CALIBRATE.

    Produces 7 points: the center and 6 scattered points around a circle of
    the given probe_radius, matching the algorithm in DeltaCalibrate.__init__.
    """
    points = [(0., 0.)]
    scatter = [.95, .90, .85, .70, .75, .80]
    for i in range(6):
        r = math.radians(90. + 60. * i)
        dist = probe_radius * scatter[i]
        points.append((math.cos(r) * dist, math.sin(r) * dist))
    return points


def generate_height_measurements(true_cal, probe_points):
    """Generate simulated probe height measurements at each probe point.

    For each (x, y) probe position the bed surface height is determined by
    the bed tilt parameters, and the corresponding stepper stable position is
    computed using the true calibration geometry.

    Returns a list of (z_offset, stable_position) tuples, where:
      z_offset        - measured Z height of the bed surface at that point
      stable_position - 3-tuple of stepper steps from endstop (tower a, b, c)
    """
    measurements = []
    for x, y in probe_points:
        z = true_cal.bed_z(x, y)
        stable_pos = true_cal.calc_stable_position([x, y, z])
        measurements.append((z, stable_pos))
    return measurements


######################################################################
# Distance measurement generation
######################################################################

def generate_distance_measurements(true_cal, scale=1.0):
    """Generate simulated distance measurements from calibration object ridges.

    Computes stable positions for the inner and outer ridge locations of the
    calibration object (docs/prints/calibrate_size.stl) using the true
    calibration, and the physical (caliper) distances between ridge pairs.

    Returns a list of (distance, stable_pos1, stable_pos2) tuples in the
    same format that DELTA_ANALYZE writes to the [delta_calibrate] section.
    The list contains 12 entries (6 center + 6 outer), matching the output
    of measurements_to_distances() in delta_calibrate.py.
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
        outer_distances.append((dist, spos1, spos2))

    return center_distances + outer_distances


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
            "section, and save them to the [delta_calibrate] section of\n"
            "the printer.cfg SAVE_CONFIG block.\n"
            "\n"
            "The [delta_true_calibration] section defines the physical\n"
            "ground-truth parameters of the delta printer:\n"
            "  delta_radius, arm_length, endstop_a/b/c,\n"
            "  angle_a/b/c, stepdist_a/b/c, bed_tilt_x, bed_tilt_y"
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
    opts = parser.parse_args()

    if not os.path.exists(opts.config):
        sys.stderr.write("Error: config file not found: %s\n" % opts.config)
        sys.exit(1)

    # Load and parse configuration
    try:
        config, orig_lines, save_config_start = parse_klipper_config(
            opts.config)
        true_cal = load_true_calibration(config)
    except (ValueError, configparser.Error) as e:
        sys.stderr.write("Error reading config: %s\n" % e)
        sys.exit(1)

    # Get probe radius from [delta_calibrate] section
    probe_radius = 50.
    if config.has_section('delta_calibrate'):
        probe_radius = config.getfloat('delta_calibrate', 'radius',
                                       fallback=50.)

    # Print summary of loaded parameters
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

    # Generate probe height measurements
    probe_points = generate_probe_points(probe_radius)
    height_measurements = generate_height_measurements(true_cal, probe_points)

    # Generate distance measurements (unless disabled)
    distance_measurements = []
    if not opts.no_distances:
        distance_measurements = generate_distance_measurements(
            true_cal, opts.scale)

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


if __name__ == '__main__':
    main()
