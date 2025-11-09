#!/usr/bin/env python3
"""
Enhanced delta calibration analysis script that can load parameters
from a Klipper configuration file.
"""

import re
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
    """
    Load delta parameters from a Klipper config file.
    
    Args:
        config_file: Path to Klipper config file (optional)
        
    Returns:
        Dictionary with delta parameters
    """
    if config_file and KLIPPER_AVAILABLE:
        try:
            print(f"Loading delta parameters from config file: {config_file}")
            config = parse_klipper_config(config_file)
            
            # Extract parameters from config
            delta_params = {}
            
            # Basic delta settings
            delta_params['radius'] = float(get_config_value(config, 'printer', 'delta_radius', '172.0'))
            
            # Arm lengths
            delta_params['arm_a'] = float(get_config_value(config, 'stepper_a', 'arm_length', '333.0'))
            delta_params['arm_b'] = float(get_config_value(config, 'stepper_b', 'arm_length', '333.0')) 
            delta_params['arm_c'] = float(get_config_value(config, 'stepper_c', 'arm_length', '333.0'))
            
            # Tower angles
            delta_params['angle_a'] = float(get_config_value(config, 'stepper_a', 'angle', '90.0'))
            delta_params['angle_b'] = float(get_config_value(config, 'stepper_b', 'angle', '210.0'))
            delta_params['angle_c'] = float(get_config_value(config, 'stepper_c', 'angle', '330.0'))
            
            # Endstop positions (from calibration)
            if config.has_section('delta_calibrate'):
                delta_params['endstop_a'] = float(get_config_value(config, 'delta_calibrate', 'delta_a', '333.0'))
                delta_params['endstop_b'] = float(get_config_value(config, 'delta_calibrate', 'delta_b', '333.0'))
                delta_params['endstop_c'] = float(get_config_value(config, 'delta_calibrate', 'delta_c', '333.0'))
            else:
                # Use position_endstop as fallback
                delta_params['endstop_a'] = float(get_config_value(config, 'stepper_a', 'position_endstop', '333.0'))
                delta_params['endstop_b'] = float(get_config_value(config, 'stepper_b', 'position_endstop', '333.0'))
                delta_params['endstop_c'] = float(get_config_value(config, 'stepper_c', 'position_endstop', '333.0'))
            
            # Stepper settings for conversion
            rotation_distance = float(get_config_value(config, 'stepper_a', 'rotation_distance', '60.0'))
            microsteps = int(get_config_value(config, 'stepper_a', 'microsteps', '32'))
            
            print("Loaded parameters from config:")
            for key, value in delta_params.items():
                print(f"  {key}: {value}")
            print(f"  rotation_distance: {rotation_distance}")
            print(f"  microsteps: {microsteps}")
            
            return delta_params, rotation_distance, microsteps\n            \n        except Exception as e:\n            print(f"Error loading config file: {e}")\n            print("Using default parameters...")\n    \n    # Default parameters if config loading fails or not specified\n    print("Using default delta parameters")\n    delta_params = {\n        'radius': 172.0,\n        'arm_a': 333.0,\n        'arm_b': 333.0,\n        'arm_c': 333.0,\n        'endstop_a': 333.0,\n        'endstop_b': 333.0,\n        'endstop_c': 333.0,\n        'angle_a': 90.0,\n        'angle_b': 210.0,\n        'angle_c': 330.0\n    }\n    return delta_params, 60.0, 32\n\ndef parse_probe_data(raw_string, rotation_distance, microsteps):\n    """\n    Parse Klipper delta_calibrate probe data and convert stepper positions to mm.\n    """\n    steps_per_mm = (200 * microsteps) / rotation_distance\n    probe_points = []\n    \n    pattern = r'#\\*# height(\\d+) = ([-+]?\\d*\\.?\\d+)\\s*\\n#\\*# height\\1_pos = ([\\d.,]+)'\n    matches = re.findall(pattern, raw_string)\n    \n    for match in matches:\n        point_num = int(match[0])\n        height = float(match[1])\n        positions_str = match[2]\n        \n        positions = [float(pos.strip()) for pos in positions_str.split(',')]\n        positions_mm = [pos / steps_per_mm for pos in positions]\n        \n        probe_point = {\n            'point_num': point_num,\n            'height': height,\n            'stable_positions_steps': positions,\n            'stable_positions_a_mm': positions_mm[0],\n            'stable_positions_b_mm': positions_mm[1],\n            'stable_positions_c_mm': positions_mm[2]\n        }\n        \n        probe_points.append(probe_point)\n    \n    return probe_points\n\n# [Rest of your existing functions would go here - stablePosition_to_carriageHeight, carriageHeight_to_cartesianNozzle, etc.]\n\n# Example usage\nif __name__ == "__main__":\n    # Try to load from config file (adjust path as needed)\n    config_file = "../../printer.cfg"  # Adjust this path to your config file\n    \n    delta_params, rotation_distance, microsteps = load_delta_params_from_config(config_file)\n    \n    # Your existing probe data\n    rawProbes = """\n#*# [delta_calibrate]\n#*# height0 = 0.0\n#*# height0_pos = 35249.010,35451.810,35413.010\n#*# height1 = 0.0\n#*# height1_pos = 42409.543,42612.343,30825.343\n# ... rest of your probe data\n"""\n    \n    # Parse probe data\n    probe_data = parse_probe_data(rawProbes, rotation_distance, microsteps)\n    \n    print(f"\\nParsed {len(probe_data)} probe points using:")\n    print(f"  Rotation distance: {rotation_distance} mm")\n    print(f"  Microsteps: {microsteps}")\n    print(f"  Steps per mm: {(200 * microsteps) / rotation_distance:.3f}")\n    \n    # Continue with your existing calibration analysis...\n