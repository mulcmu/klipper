#!/usr/bin/env python3
"""
Enhanced delta calibration analysis script that can load parameters
from a Klipper configuration file.
"""

import numpy as np
from scipy.optimize import minimize
import math
import sys
import os
import configparser

# Add the Klipper path so we can import its modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    from klippy import configfile
    KLIPPER_AVAILABLE = True
except ImportError:
    print("Warning: Klipper modules not available. Using manual config only.")
    KLIPPER_AVAILABLE = False

def parse_klipper_config(config_filename):
    """Parse a Klipper configuration file using Klipper's built-in parser."""
    if not KLIPPER_AVAILABLE:
        raise ImportError("Klipper modules not available")

    config_reader = configfile.ConfigFileReader()
    config_data = config_reader.read_config_file(config_filename)
    fileconfig = config_reader._create_fileconfig()
    config_reader.append_fileconfig(fileconfig, config_data, config_filename)
    return fileconfig

def get_config_value(fileconfig, section, option, fallback=None):
    """Get a configuration value with proper error handling."""
    try:
        return fileconfig.get(section, option)
    except (configparser.NoSectionError, configparser.NoOptionError):
        return fallback

def load_delta_params_from_config(config_file=None):
    """Load delta parameters from a Klipper config file."""
    if config_file and KLIPPER_AVAILABLE:
        try:
            print(f"Loading delta parameters from config file: {config_file}")
            config = parse_klipper_config(config_file)

            delta_params = {}

            # Tower radii with inheritance
            radius_a = float(get_config_value(config, 'stepper_a', 'radius', '172.0'))
            radius_b_raw = get_config_value(config, 'stepper_b', 'radius', None)
            radius_c_raw = get_config_value(config, 'stepper_c', 'radius', None)
            radius_b = float(radius_b_raw) if radius_b_raw is not None else radius_a
            radius_c = float(radius_c_raw) if radius_c_raw is not None else radius_a
            delta_params['radius_a'] = radius_a
            delta_params['radius_b'] = radius_b
            delta_params['radius_c'] = radius_c

            # Arm lengths with inheritance
            arm_a = float(get_config_value(config, 'stepper_a', 'arm_length', '333.0'))
            arm_b_raw = get_config_value(config, 'stepper_b', 'arm_length', None)
            arm_c_raw = get_config_value(config, 'stepper_c', 'arm_length', None)
            arm_b = float(arm_b_raw) if arm_b_raw is not None else arm_a
            arm_c = float(arm_c_raw) if arm_c_raw is not None else arm_a
            delta_params['arm_a'] = arm_a
            delta_params['arm_b'] = arm_b
            delta_params['arm_c'] = arm_c

            # Tower angles
            delta_params['angle_a'] = float(get_config_value(config, 'stepper_a', 'angle', '90.0'))
            delta_params['angle_b'] = float(get_config_value(config, 'stepper_b', 'angle', '210.0'))
            delta_params['angle_c'] = float(get_config_value(config, 'stepper_c', 'angle', '330.0'))


            # Endstop positions come from individual stepper sections
            endstop_a_raw = get_config_value(config, 'stepper_a', 'position_endstop', '333.0')
            endstop_b_raw = get_config_value(config, 'stepper_b', 'position_endstop', endstop_a_raw)
            endstop_c_raw = get_config_value(config, 'stepper_c', 'position_endstop', endstop_a_raw)
            delta_params['endstop_a'] = float(endstop_a_raw)
            delta_params['endstop_b'] = float(endstop_b_raw)
            delta_params['endstop_c'] = float(endstop_c_raw)

            # Stepper settings for conversion
            rotation_distance = float(get_config_value(config, 'stepper_a', 'rotation_distance', '60.0'))
            microsteps = int(get_config_value(config, 'stepper_a', 'microsteps', '32'))

            print("Loaded parameters from config:")
            for key, value in delta_params.items():
                print(f"  {key}: {value}")
            print(f"  rotation_distance: {rotation_distance}")
            print(f"  microsteps: {microsteps}")

            return delta_params, rotation_distance, microsteps
        except Exception as e:
            print(f"Error loading config file: {e}")
            print("Using default parameters...")

    # Default parameters if config loading fails or not specified
    print("Using default delta parameters")
    delta_params = {
        'radius_a': 172.0,
        'radius_b': 172.0,
        'radius_c': 172.0,
        'arm_a': 333.0,
        'arm_b': 333.0,
        'arm_c': 333.0,
        'endstop_a': 333.0,
        'endstop_b': 333.0,
        'endstop_c': 333.0,
        'angle_a': 90.0,
        'angle_b': 210.0,
        'angle_c': 330.0,
    }
    return delta_params, 60.0, 32

def _extract_autosave_fileconfig(config_path):
    """Return a RawConfigParser built from the SAVE_CONFIG block."""
    cfgrdr = configfile.ConfigFileReader()
    try:
        data = cfgrdr.read_config_file(config_path)
    except Exception as err:
        print(f"Unable to read config file '{config_path}': {err}")
        return None

    header = configfile.AUTOSAVE_HEADER
    pos = data.find(header)
    if pos < 0:
        return None

    autosave_raw = data[pos + len(header):]
    autosave_lines = []
    for line in autosave_raw.splitlines():
        if not line.startswith('#*#'):
            break
        stripped = line[4:]
        autosave_lines.append(stripped[1:] if stripped.startswith(' ') else stripped)

    autosave_data = '\n'.join(autosave_lines).strip()
    if not autosave_data:
        return None

    return cfgrdr.build_fileconfig(autosave_data, config_path)


def _load_autosave_section(config_file):
    if not config_file or not os.path.exists(config_file):
        print("No config file found; returning empty autosave data.")
        return None
    autosave_cfg = _extract_autosave_fileconfig(config_file)
    if autosave_cfg is None or not autosave_cfg.has_section('delta_calibrate'):
        print("No saved [delta_calibrate] data found in config; returning empty autosave data.")
        return None
    return autosave_cfg


def _parse_stable_position(autosave_cfg, section, key):
    if not autosave_cfg.has_option(section, key):
        return None
    positions_str = autosave_cfg.get(section, key)
    try:
        positions = [float(part.strip()) for part in positions_str.split(',')]
    except ValueError:
        return None
    if len(positions) != 3:
        return None
    return positions


def parse_probe_data(config_file, rotation_distance, microsteps, autosave_cfg=None):
    """Parse delta_calibrate probe heights using Klipper's autosave data."""
    steps_per_mm = (200 * microsteps) / rotation_distance
    probe_points = []

    autosave_cfg = autosave_cfg or _load_autosave_section(config_file)
    if autosave_cfg is None:
        return probe_points

    section = 'delta_calibrate'
    idx = 0
    while autosave_cfg.has_option(section, f'height{idx}'):
        height = autosave_cfg.getfloat(section, f'height{idx}')
        positions = _parse_stable_position(autosave_cfg, section, f'height{idx}_pos')
        if positions is None:
            idx += 1
            continue

        positions_mm = [pos / steps_per_mm for pos in positions]
        probe_points.append({
            'point_num': idx,
            'height': height,
            'stable_positions_steps': positions,
            'stable_positions_a_mm': positions_mm[0],
            'stable_positions_b_mm': positions_mm[1],
            'stable_positions_c_mm': positions_mm[2],
        })
        idx += 1

    return probe_points


def parse_distance_data(config_file, rotation_distance, microsteps, autosave_cfg=None):
    """Parse delta_calibrate distance measurements from autosave data."""
    steps_per_mm = (200 * microsteps) / rotation_distance
    distance_points = []

    autosave_cfg = autosave_cfg or _load_autosave_section(config_file)
    if autosave_cfg is None:
        return distance_points

    section = 'delta_calibrate'
    idx = 0
    while autosave_cfg.has_option(section, f'distance{idx}'):
        distance = autosave_cfg.getfloat(section, f'distance{idx}')
        pos1 = _parse_stable_position(autosave_cfg, section, f'distance{idx}_pos1')
        pos2 = _parse_stable_position(autosave_cfg, section, f'distance{idx}_pos2')
        if pos1 is None or pos2 is None:
            idx += 1
            continue

        pos1_mm = [pos / steps_per_mm for pos in pos1]
        pos2_mm = [pos / steps_per_mm for pos in pos2]
        distance_points.append({
            'point_num': idx,
            'distance': distance,
            'stable_pos1_steps': pos1,
            'stable_pos1_a_mm': pos1_mm[0],
            'stable_pos1_b_mm': pos1_mm[1],
            'stable_pos1_c_mm': pos1_mm[2],
            'stable_pos2_steps': pos2,
            'stable_pos2_a_mm': pos2_mm[0],
            'stable_pos2_b_mm': pos2_mm[1],
            'stable_pos2_c_mm': pos2_mm[2],
        })
        idx += 1

    return distance_points

# [Rest of your existing functions would go here - stablePosition_to_carriageHeight, carriageHeight_to_cartesianNozzle, etc.]

# Example usage
if __name__ == "__main__":
    # Try to load from config file (pass path as first CLI argument)
    default_config = os.path.join(os.path.dirname(__file__), 'printer.cfg')
    config_file = sys.argv[1] if len(sys.argv) > 1 else default_config
    config_file = os.path.abspath(config_file)

    delta_params, rotation_distance, microsteps = load_delta_params_from_config(config_file)

    autosave_cfg = _load_autosave_section(config_file)

    # Parse probe and distance data saved in the config file
    probe_data = parse_probe_data(config_file, rotation_distance, microsteps, autosave_cfg=autosave_cfg)
    distance_data = parse_distance_data(config_file, rotation_distance, microsteps, autosave_cfg=autosave_cfg)

    print(f"\nParsed {len(probe_data)} probe points using:")
    print(f"  Rotation distance: {rotation_distance} mm")
    print(f"  Microsteps: {microsteps}")
    print(f"  Steps per mm: {(200 * microsteps) / rotation_distance:.3f}")
    print(f"Found {len(distance_data)} distance measurements in autosave data.")
    
    print(probe_data)
    print(distance_data)

    # Continue with your existing calibration analysis...
