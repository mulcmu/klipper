# Simple math helper functions
#
# Copyright (C) 2018-2019  Kevin O'Connor <kevin@koconnor.net>
#
# This file may be distributed under the terms of the GNU GPLv3 license.
import math, logging, multiprocessing, traceback
import queuelogger
import random


######################################################################
# Coordinate descent
######################################################################

# Helper code that implements coordinate descent
def coordinate_descent(adj_params, params, error_func, gcode=None):
    params = dict(params)
    best_err = error_func(params)
    if gcode:
        gcode.respond_info("Coordinate descent initial error: %s" % (best_err,))
    else:
        logging.info("Coordinate descent initial error: %s", best_err)

    height_candidates = [
        'endstop_a', 'endstop_b', 'endstop_c',
        'radius', 'tilt_x', 'tilt_y']
    dist_candidates = [
        'arm_a', 'arm_b', 'arm_c',
        'angle_a', 'angle_b', 'angle_c']
    
    height_params = [p for p in height_candidates
                     if p in params and p in adj_params]
    dist_params = [p for p in dist_candidates
                   if p in params and p in adj_params] + height_params

    best_val = 20 * (best_err[0] * best_err[1]) + (best_err[0] - best_err[1])**2 + 20 * best_err[1]
    threshold = 0.00001
    for cycle in range(5):
        random.shuffle(height_params)
        random.shuffle(dist_params)

        rounds = 0
        dp = {param_name: 1. for param_name in dist_params}
        while dp and sum(dp.values()) > threshold and rounds < 500:
            rounds += 1
            for param_name in dist_params:
                orig = params[param_name]
                params[param_name] = orig + dp[param_name]
                err = error_func(params)
                if best_val > 20 *(err[0] * err[1]) + (err[0] - err[1])**2 + 20 * err[1]:
                    best_err = err
                    best_val = 20 *(err[0] * err[1]) + (err[0] - err[1])**2 + 20 * err[1]
                    dp[param_name] *= 1.1
                    continue
                params[param_name] = orig - dp[param_name]
                err = error_func(params)
                if best_val > 20 *(err[0] * err[1]) + (err[0] - err[1])**2 + 20 * err[1]:
                    best_err = err
                    best_val = 20 *(err[0] * err[1]) + (err[0] - err[1])**2 + 20 * err[1]
                    dp[param_name] *= 1.1
                    continue
                params[param_name] = orig
                dp[param_name] *= 0.9
            if gcode and rounds % 25 == 0:
                gcode.respond_info(
                    "Delta error: %s  Cycle: D%d  Rounds: %d" % (best_err, cycle, rounds))
        if gcode:
            gcode.respond_info(str(params))
            
        rounds = 0        
        dp = {param_name: 1. for param_name in height_params}
        while dp and sum(dp.values()) > threshold and rounds < 500:
            rounds += 1
            for param_name in height_params:
                orig = params[param_name]
                params[param_name] = orig + dp[param_name]
                err = error_func(params)
                if best_val > 20 *(err[0] * err[1]) + (err[0] - err[1])**2 + 20 * err[1]:
                    best_err = err
                    best_val = 20 *(err[0] * err[1]) + (err[0] - err[1])**2 + 20 * err[1]
                    best_h = error_func(params)
                    dp[param_name] *= 1.1
                    continue
                params[param_name] = orig - dp[param_name]
                err = error_func(params)
                if best_val > 20 *(err[0] * err[1]) + (err[0] - err[1])**2 + 20 * err[1]:
                    best_err = err
                    best_val = 20 *(err[0] * err[1]) + (err[0] - err[1])**2 + 20 * err[1]
                    dp[param_name] *= 1.1
                    continue
                params[param_name] = orig
                dp[param_name] *= 0.9
            if gcode and rounds % 25 == 0:
                gcode.respond_info(
                    "Delta error: %s  Cycle: H%d  Rounds: %d" % (best_err, cycle, rounds))
        if gcode:
            gcode.respond_info(str(params))
            
    final_rounds = rounds
    if gcode:
        gcode.respond_info(
            "Coordinate descent best_err: %s  rounds: %d" % (best_err, final_rounds))
    else:
        logging.info("Coordinate descent best_err: %s  rounds: %d",
                     best_err, final_rounds)
    return params

# Helper to run the coordinate descent function in a background
# process so that it does not block the main thread.
def background_coordinate_descent(printer, adj_params, params, error_func):
    parent_conn, child_conn = multiprocessing.Pipe()
    gcode = printer.lookup_object("gcode")
    def wrapper():
        queuelogger.clear_bg_logging()
        try:
            res = coordinate_descent(adj_params, params, error_func, gcode)
        except:
            child_conn.send((True, traceback.format_exc()))
            child_conn.close()
            return
        child_conn.send((False, res))
        child_conn.close()
    # Start a process to perform the calculation
    calc_proc = multiprocessing.Process(target=wrapper)
    calc_proc.daemon = True
    calc_proc.start()
    # Wait for the process to finish
    reactor = printer.get_reactor()
    gcode = printer.lookup_object("gcode")
    eventtime = last_report_time = reactor.monotonic()
    while calc_proc.is_alive():
        if eventtime > last_report_time + 5.:
            last_report_time = eventtime
            gcode.respond_info("Working on calibration...", log=False)
        eventtime = reactor.pause(eventtime + .1)
    # Return results
    is_err, res = parent_conn.recv()
    if is_err:
        raise Exception("Error in coordinate descent: %s" % (res,))
    calc_proc.join()
    parent_conn.close()
    return res


######################################################################
# Trilateration
######################################################################

# Trilateration finds the intersection of three spheres.  See the
# wikipedia article for the details of the algorithm.
def trilateration(sphere_coords, radius2):
    sphere_coord1, sphere_coord2, sphere_coord3 = sphere_coords
    s21 = matrix_sub(sphere_coord2, sphere_coord1)
    s31 = matrix_sub(sphere_coord3, sphere_coord1)

    d = math.sqrt(matrix_magsq(s21))
    ex = matrix_mul(s21, 1. / d)
    i = matrix_dot(ex, s31)
    vect_ey = matrix_sub(s31, matrix_mul(ex, i))
    ey = matrix_mul(vect_ey, 1. / math.sqrt(matrix_magsq(vect_ey)))
    ez = matrix_cross(ex, ey)
    j = matrix_dot(ey, s31)

    x = (radius2[0] - radius2[1] + d**2) / (2. * d)
    y = (radius2[0] - radius2[2] - x**2 + (x-i)**2 + j**2) / (2. * j)
    z = -math.sqrt(radius2[0] - x**2 - y**2)

    ex_x = matrix_mul(ex, x)
    ey_y = matrix_mul(ey, y)
    ez_z = matrix_mul(ez, z)
    return matrix_add(sphere_coord1, matrix_add(ex_x, matrix_add(ey_y, ez_z)))


######################################################################
# Matrix helper functions for 3x1 matrices
######################################################################

def matrix_cross(m1, m2):
    return [m1[1] * m2[2] - m1[2] * m2[1],
            m1[2] * m2[0] - m1[0] * m2[2],
            m1[0] * m2[1] - m1[1] * m2[0]]

def matrix_dot(m1, m2):
    return m1[0] * m2[0] + m1[1] * m2[1] + m1[2] * m2[2]

def matrix_magsq(m1):
    return m1[0]**2 + m1[1]**2 + m1[2]**2

def matrix_add(m1, m2):
    return [m1[0] + m2[0], m1[1] + m2[1], m1[2] + m2[2]]

def matrix_sub(m1, m2):
    return [m1[0] - m2[0], m1[1] - m2[1], m1[2] - m2[2]]

def matrix_mul(m1, s):
    return [m1[0]*s, m1[1]*s, m1[2]*s]

######################################################################
# Matrix helper functions for 3x3 matrices
######################################################################

def matrix_det(a):
    x0, x1, x2 = a
    return matrix_dot(x0, matrix_cross(x1, x2))

def matrix_inv(a):
    x0, x1, x2 = a
    inv_det = 1. / matrix_det(a)
    return [matrix_mul(matrix_cross(x1, x2), inv_det),
            matrix_mul(matrix_cross(x2, x0), inv_det),
            matrix_mul(matrix_cross(x0, x1), inv_det)]




    # params = dict(params)
    # best_err = error_func(params)
    # if gcode:
    #     gcode.respond_info("Coordinate descent initial error: %s" % (best_err,))
    # else:
    #     logging.info("Coordinate descent initial error: %s", best_err)

    # height_candidates = [
    #     'endstop_a', 'endstop_b', 'endstop_c',
    #     'radius', 'tilt_x', 'tilt_y']
    # dist_candidates = [
    #     'arm_a', 'arm_b', 'arm_c',
    #     'angle_a', 'angle_b', 'angle_c']
    
    # height_params = [p for p in height_candidates
    #                  if p in params and p in adj_params]
    # dist_params = [p for p in dist_candidates
    #                if p in params and p in adj_params] + height_params

    # threshold = 0.00001
    # for cycle in range(5):
    #     random.shuffle(height_params)
    #     random.shuffle(dist_params)

    #     rounds = 0
    #     dp = {param_name: 1. for param_name in dist_params}
    #     while dp and sum(dp.values()) > threshold and rounds < 500:
    #         rounds += 1
    #         for param_name in dist_params:
    #             orig = params[param_name]
    #             params[param_name] = orig + dp[param_name]
    #             err = error_func(params)
    #             if err[1] < best_err[1]:
    #                 best_err = err
    #                 best_d = error_func(params)
    #                 dp[param_name] *= 1.1
    #                 continue
    #             params[param_name] = orig - dp[param_name]
    #             err = error_func(params)
    #             if err[1] < best_err[1]:
    #                 best_err = err
    #                 best_d = error_func(params)
    #                 dp[param_name] *= 1.1
    #                 continue
    #             params[param_name] = orig
    #             dp[param_name] *= 0.9
    #         if gcode and rounds % 25 == 0:
    #             gcode.respond_info(
    #                 "Delta error: %s  Cycle: D%d  Rounds: %d" % (best_err, cycle, rounds))
    #     if gcode:
    #         gcode.respond_info(str(params))
            
    #     rounds = 0        
    #     dp = {param_name: 1. for param_name in height_params}
    #     while dp and sum(dp.values()) > threshold and rounds < 500:
    #         rounds += 1
    #         for param_name in height_params:
    #             orig = params[param_name]
    #             params[param_name] = orig + dp[param_name]
    #             err = error_func(params)
    #             if err[0] < best_err[0]:
    #                 best_err = err
    #                 best_h = error_func(params)
    #                 dp[param_name] *= 1.1
    #                 continue
    #             params[param_name] = orig - dp[param_name]
    #             err = error_func(params)
    #             if err[0] < best_err[0]:
    #                 best_err = err
    #                 best_h = error_func(params)
    #                 dp[param_name] *= 1.1
    #                 continue
    #             params[param_name] = orig
    #             dp[param_name] *= 0.9
    #         if gcode and rounds % 25 == 0:
    #             gcode.respond_info(
    #                 "Delta error: %s  Cycle: H%d  Rounds: %d" % (best_err, cycle, rounds))
    #     if gcode:
    #         gcode.respond_info(str(params))
    #         # gcode.respond_info(f"Best height error: {best_h}  Best distance error: {best_d}")