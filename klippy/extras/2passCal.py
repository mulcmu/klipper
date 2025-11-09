import re
import numpy as np
from scipy.optimize import minimize
import math

microsteps = 32
rotation_distance = 60

rawProbes = """
#*# [delta_calibrate]
#*# height0 = 0.0
#*# height0_pos = 35249.010,35451.810,35413.010
#*# height1 = 0.0
#*# height1_pos = 42409.543,42612.343,30825.343
#*# height2 = 0.0
#*# height2_pos = 38621.623,38823.623,32034.623
#*# height3 = 0.0
#*# height3_pos = 36643.932,36845.932,33461.732
#*# height4 = 0.0
#*# height4_pos = 37934.017,46521.217,31706.217
#*# height5 = 0.0
#*# height5_pos = 36196.452,40739.652,32578.652
#*# height6 = 0.0
#*# height6_pos = 35486.159,37721.159,33745.159
#*# height7 = 0.0
#*# height7_pos = 34083.898,48164.898,34247.898
#*# height8 = 0.0
#*# height8_pos = 33950.303,41471.503,34114.303
#*# height9 = 0.0
#*# height9_pos = 34369.867,38046.867,34533.867
#*# height10 = 0.0
#*# height10_pos = 31524.904,46503.904,38079.904
#*# height11 = 0.0
#*# height11_pos = 32403.353,40728.353,36348.353
#*# height12 = 0.0
#*# height12_pos = 33577.758,37716.958,35645.958
#*# height13 = 0.0
#*# height13_pos = 30650.048,42600.048,42561.048
#*# height14 = 0.0
#*# height14_pos = 31859.627,38811.627,38773.427
#*# height15 = 0.0
#*# height15_pos = 33291.709,36839.709,36800.709
#*# height16 = 0.0
#*# height16_pos = 31550.086,38144.086,46490.086
#*# height17 = 0.0
#*# height17_pos = 32411.013,36395.013,40697.013
#*# height18 = 0.0
#*# height18_pos = 33577.960,35684.960,37678.960
#*# height19 = 0.0
#*# height19_pos = 34109.341,34311.341,48151.341
#*# height20 = 0.0
#*# height20_pos = 33956.461,34158.461,41439.261
#*# height21 = 0.0
#*# height21_pos = 34371.136,34573.336,38008.336
#*# height22 = 0.0
#*# height22_pos = 37932.140,31743.140,46481.140
#*# height23 = 0.0
#*# height23_pos = 36192.367,32613.367,40696.567
#*# height24 = 0.0
#*# height24_pos = 35482.859,33780.059,37679.059
#*# height25 = 0.0
#*# height25_pos = 42396.985,30850.985,42559.985
#*# height26 = 0.0
#*# height26_pos = 38610.651,32062.651,38774.651
#*# height27 = 0.0
#*# height27_pos = 36636.840,33493.640,36800.640
#*# height28 = 0.0
#*# height28_pos = 46310.262,31736.262,38089.062
#*# height29 = 0.0
#*# height29_pos = 40529.096,32609.096,36352.096
#*# height30 = 0.0
#*# height30_pos = 37515.315,33780.315,35646.315
#*# height31 = 0.0
#*# height31_pos = 47955.706,34279.706,34240.706
#*# height32 = 0.0
#*# height32_pos = 41272.925,34156.125,34117.125
#*# height33 = 0.0
#*# height33_pos = 37842.223,34570.223,34531.223
#*# height34 = 0.0
#*# height34_pos = 46315.246,38132.246,31702.246
#*# height35 = 0.0
#*# height35_pos = 40536.725,36397.725,32577.725
#*# height36 = 0.0
#*# height36_pos = 37519.223,35688.223,33745.223
#*# height37 = 0.0
#*# height37_pos = 35827.095,34921.095,35431.095
#*# height38 = 0.0
#*# height38_pos = 34716.920,36027.920,35429.120
#*# height39 = 0.0
#*# height39_pos = 35589.661,35791.661,34797.661
#*# height40 = 0.0
#*# height40_pos = 34948.106,35150.906,36077.106
#*#"""




def parse_probe_data(raw_string, rotation_distance, microsteps):
    """
    Parse Klipper delta_calibrate probe data and convert stepper positions to mm.
    
    Args:
        raw_string: Raw probe data string from Klipper config
        rotation_distance: Rotation distance in mm per full rotation
        microsteps: Number of microsteps per full step (typically 16 or 32)
        
    Returns:
        List of dictionaries with stable probe point data in mm
    """
    # Calculate conversion factor from stepper positions to mm
    # Each full step moves rotation_distance mm
    # Each microstep is 1/microsteps of a full step
    steps_per_mm = (200 * microsteps) / rotation_distance  # 200 full steps per rotation
    
    # Parse the probe data
    probe_points = []
    
    # Find all height entries with positions
    pattern = r'#\*# height(\d+) = ([-+]?\d*\.?\d+)\s*\n#\*# height\1_pos = ([\d.,]+)'
    
    matches = re.findall(pattern, raw_string)
    
    for match in matches:
        point_num = int(match[0])
        height = float(match[1])
        positions_str = match[2]
        
        # Parse the three stepper positions (A, B, C towers)
        positions = [float(pos.strip()) for pos in positions_str.split(',')]
        
        # Convert stepper positions to mm
        # Note: These are "stable positions" - steps from endstop
        positions_mm = [pos / steps_per_mm for pos in positions]
        
        probe_point = {
            'point_num': point_num,
            'height': height,
            'stable_positions_steps': positions,
            'stable_positions_a_mm': positions_mm[0],
            'stable_positions_b_mm': positions_mm[1],
            'stable_positions_c_mm': positions_mm[2]
        }
        
        probe_points.append(probe_point)
    
    return probe_points

def stablePosition_to_carriageHeight(stable_positions_mm, delta_params):
    """
    Convert stable positions in mm to carriage heights in mm using delta kinematics.
    
    For delta printers, the stable position is the distance traveled from the endstop trigger.
    The carriage height is simply: endstop_height + stable_position
    
    Args:
        stable_positions_mm: List of three stable positions [A, B, C] in mm
        delta_params: Dictionary with delta parameters:
            - 'endstop_a', 'endstop_b', 'endstop_c': Endstop heights (mm)

    Returns:
        List of three carriage heights [Z_a, Z_b, Z_c] in mm
    """
    # Unpack stable positions
    a_mm, b_mm, c_mm = stable_positions_mm

    # Unpack delta parameters
    endstop_a = delta_params['endstop_a']
    endstop_b = delta_params['endstop_b']
    endstop_c = delta_params['endstop_c']
    arm_a = delta_params['arm_a']
    arm_b = delta_params['arm_b']
    arm_c = delta_params['arm_c']
    radius = delta_params['radius']

    # Calculate carriage heights using proper delta geometry
    # When endstop triggers, nozzle is at endstop_height above bed
    # The vertical offset due to arm geometry is sqrt(arm² - radius²)
    # As carriage moves down by stable_position, carriage height decreases
    z_a = endstop_a + np.sqrt(arm_a**2 - radius**2) - a_mm
    z_b = endstop_b + np.sqrt(arm_b**2 - radius**2) - b_mm
    z_c = endstop_c + np.sqrt(arm_c**2 - radius**2) - c_mm

    return [z_a, z_b, z_c]

def carriageHeight_to_cartesianNozzle(carriage_heights_mm, delta_params):
    """
    Convert carriage heights in mm to Cartesian coordinates (X, Y, Z) in mm using spherical
    intersections, trilateration.
    
    Args:
        carriage_heights_mm: List of three carriage heights [Z_a, Z_b, Z_c] in mm
        delta_params: Dictionary with delta parameters:
            - 'radius': Horizontal distance from center to tower (mm)
            - 'arm_a', 'arm_b', 'arm_c': Lengths of the arms (mm)
            - 'endstop_a', 'endstop_b', 'endstop_c': Endstop offsets (mm)
            - 'angle_a', 'angle_b', 'angle_c': Angles of the towers (degrees)

    Returns:
        List of Cartesian coordinates [X, Y, Z] in mm
    """
    # Unpack carriage heights
    z_a, z_b, z_c = carriage_heights_mm

    # Unpack delta parameters
    radius = delta_params['radius']
    arm_a = delta_params['arm_a']
    arm_b = delta_params['arm_b']
    arm_c = delta_params['arm_c']
    angle_a = np.radians(delta_params['angle_a'])
    angle_b = np.radians(delta_params['angle_b'])
    angle_c = np.radians(delta_params['angle_c'])

    # Tower positions in XY plane
    ax, ay = radius * np.cos(angle_a), radius * np.sin(angle_a)
    bx, by = radius * np.cos(angle_b), radius * np.sin(angle_b)
    cx, cy = radius * np.cos(angle_c), radius * np.sin(angle_c) 
    
    # For delta printer forward kinematics, we have three spheres centered at 
    # tower positions (ax, ay, z_a), (bx, by, z_b), (cx, cy, z_c) with radii arm_a, arm_b, arm_c
    # We need to find their intersection point (x, y, z)
    
    # Using standard trilateration approach for three spheres:
    # (x - ax)² + (y - ay)² + (z - z_a)² = arm_a²
    # (x - bx)² + (y - by)² + (z - z_b)² = arm_b²
    # (x - cx)² + (y - cy)² + (z - z_c)² = arm_c²
    
    # Expand and subtract first equation from second and third:
    # Equation 1: A*x + B*y + C*z = D
    # Equation 2: E*x + F*y + G*z = H
    
    # From sphere A to sphere B
    A1 = 2 * (bx - ax)
    B1 = 2 * (by - ay)
    C1 = 2 * (z_b - z_a)
    D1 = (arm_b**2 - arm_a**2) - (bx**2 - ax**2) - (by**2 - ay**2) - (z_b**2 - z_a**2)
    
    # From sphere A to sphere C  
    A2 = 2 * (cx - ax)
    B2 = 2 * (cy - ay)
    C2 = 2 * (z_c - z_a)
    D2 = (arm_c**2 - arm_a**2) - (cx**2 - ax**2) - (cy**2 - ay**2) - (z_c**2 - z_a**2)
    
    # If C1 and C2 are both near zero, solve directly for x,y
    if abs(C1) < 1e-6 and abs(C2) < 1e-6:
        # Solve 2x2 system: A1*x + B1*y = D1, A2*x + B2*y = D2
        denom = A1 * B2 - B1 * A2
        if abs(denom) < 1e-6:
            raise ValueError("Degenerate case: towers are collinear or equations are dependent.")
        x = (D1 * B2 - B1 * D2) / denom
        y = (A1 * D2 - D1 * A2) / denom
        
        # Calculate z from first sphere equation
        z_squared = arm_a**2 - (x - ax)**2 - (y - ay)**2
        if z_squared < 0:
            # Point is outside reachable workspace
            horizontal_distance = np.sqrt((x - ax)**2 + (y - ay)**2)
            print(f"Warning: Point outside workspace (case 1). Horizontal distance to tower A: {horizontal_distance:.2f}mm, arm length: {arm_a:.2f}mm")
            print(f"  Calculated position: X={x:.2f}, Y={y:.2f}, carriage heights: A={z_a:.2f}, B={z_b:.2f}, C={z_c:.2f}")
            # Return an approximate Z value
            z = (z_a + z_b + z_c) / 3.0 - arm_a  # Rough estimate
        else:
            z = z_a - np.sqrt(z_squared)  # Take lower solution (bed is below carriages)
        
    else:
        # General case: solve for z first, then x,y
        # From the two plane equations, eliminate x and y to get z
        
        # If C1 ≠ 0, express z from first equation: z = (D1 - A1*x - B1*y) / C1
        # Substitute into second equation and solve
        
        if abs(C1) > 1e-6:
            # Use equation 1 to express z in terms of x,y: z = (D1 - A1*x - B1*y) / C1
            # Substitute into equation 2: A2*x + B2*y + C2*(D1 - A1*x - B1*y)/C1 = D2
            # Rearrange: (A2 - C2*A1/C1)*x + (B2 - C2*B1/C1)*y = D2 - C2*D1/C1
            
            A_eff = A2 - C2 * A1 / C1
            B_eff = B2 - C2 * B1 / C1  
            D_eff = D2 - C2 * D1 / C1
            
            # This gives us one equation in x,y. We need another constraint.
            # Use the original sphere equation A: (x - ax)² + (y - ay)² + (z - z_a)² = arm_a²
            # Substitute z = (D1 - A1*x - B1*y) / C1
            
            # This becomes a quadratic equation. For simplicity, let's use a different approach:
            # Solve the linear system A1*x + B1*y + C1*z = D1, A2*x + B2*y + C2*z = D2
            # along with the constraint from sphere A
            
            # Use iterative approach or solve analytically
            # For now, let's use the standard delta kinematics approach:
            
            # Calculate using the method from RepRap firmware / Klipper
            # This is the standard forward kinematics for delta printers
            
            # Intermediate variables for cleaner calculation
            xa2 = ax * ax
            ya2 = ay * ay  
            xb2 = bx * bx
            yb2 = by * by
            xc2 = cx * cx
            yc2 = cy * cy
            
            za2 = z_a * z_a
            zb2 = z_b * z_b
            zc2 = z_c * z_c
            
            L1_2 = arm_a * arm_a
            L2_2 = arm_b * arm_b
            L3_2 = arm_c * arm_c
            
            # System of linear equations coefficients
            dnm = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
            
            if abs(dnm) < 1e-6:
                raise ValueError("Degenerate case: towers are collinear.")
            
            w1 = xa2 + ya2 + za2
            w2 = xb2 + yb2 + zb2
            w3 = xc2 + yc2 + zc2
            
            a1 = (za2 - zb2) + (w1 - w2) + (L2_2 - L1_2)
            b1 = (zb2 - zc2) + (w2 - w3) + (L3_2 - L2_2)
            
            # Solve for x and y
            x = (a1 * (by - cy) - b1 * (ay - by)) / dnm
            y = (b1 * (ax - bx) - a1 * (cx - bx)) / dnm
            
            # Solve for z using constraint from tower A
            z_discriminant = L1_2 - (x - ax)**2 - (y - ay)**2
            
            if z_discriminant < 0:
                # Point is outside reachable workspace - this can happen with calibration errors
                # Return the approximate position but mark it as invalid
                horizontal_distance = np.sqrt((x - ax)**2 + (y - ay)**2)
                print(f"Warning: Point outside workspace. Horizontal distance to tower A: {horizontal_distance:.2f}mm, arm length: {arm_a:.2f}mm")
                print(f"  Calculated position: X={x:.2f}, Y={y:.2f}, carriage heights: A={z_a:.2f}, B={z_b:.2f}, C={z_c:.2f}")
                # Return an approximate Z value based on the average carriage height
                z = (z_a + z_b + z_c) / 3.0 - arm_a  # Rough estimate
            else:
                z = z_a - np.sqrt(z_discriminant)  # Take the lower solution
        
        else:
            raise ValueError("Unsupported configuration: C1 is zero but C2 is not.")
    
    return [x, y, z]

delta_params = {
    'radius': 172.0,      # Example radius in mm
    'arm_a': 340.0,       # Example arm length A in mm
    'arm_b': 340.0,       # Example arm length B in mm
    'arm_c': 340.0,       # Example arm length C in mm
    'endstop_a': 333.0,     # Example endstop offset A in mm
    'endstop_b': 333.0,     # Example endstop offset B in mm
    'endstop_c': 333.0,     # Example endstop offset C in mm
    'angle_a': 90.0,       # Angle A in degrees
    'angle_b': 210.0,     # Angle B in degrees
    'angle_c': 330.0      # Angle C in degrees
}

# Parse the probe data
probe_data = parse_probe_data(rawProbes, rotation_distance, microsteps)

# Print the parsed probe data
for point in probe_data:
    print(point)

carriage_heights = []
for point in probe_data:
    stable_positions_mm = [
        point['stable_positions_a_mm'],
        point['stable_positions_b_mm'],
        point['stable_positions_c_mm']
    ]
    sp = stablePosition_to_carriageHeight(stable_positions_mm, delta_params)
    carriage_heights.append(sp)
    print(f"Point {point['point_num']} Carriage Heights (mm): {sp}")

for point in carriage_heights:
    x,y,z = carriageHeight_to_cartesianNozzle(point, delta_params)
    print(f"Cartesian Coordinates (mm): X={x:.2f}, Y={y:.2f}, Z={z:.2f}")
    
