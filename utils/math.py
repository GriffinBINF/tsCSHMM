import numpy as np

def cubic_bezier(P0, P1, P2, P3, t):
    return (1 - t)[:, None] ** 3 * P0 + \
           3 * (1 - t)[:, None] ** 2 * t[:, None] * P1 + \
           3 * (1 - t)[:, None] * t[:, None] ** 2 * P2 + \
           t[:, None] ** 3 * P3

def cubic_bezier_derivative(P0, P1, P2, P3, t):
    return 3 * (1 - t)[:, None] ** 2 * (P1 - P0) + \
           6 * (1 - t)[:, None] * t[:, None] * (P2 - P1) + \
           3 * t[:, None] ** 2 * (P3 - P2)
