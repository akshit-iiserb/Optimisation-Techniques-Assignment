import numpy as np
from scipy.optimize import linprog

def get_params(roll="25027"):
    digits = [c for c in str(roll) if c.isdigit()]
    d1, d2, d3, d4 = map(int, digits[-4:])
    p = d4 % 5 + 1
    q = d3 % 4 + 1
    r = d2 % 3 + 1
    k = (d1 + d2 + d3 + d4) % 9 + 1
    return p, q, r, k

def run(roll="25027"):
    p, q, r, k = get_params(roll)
    print(f"Your roll number: {roll}   p = {p}   q = {q}   r = {r}   k = {k}")
    
    # 1. Primal LP (max z => min -z)
    c_primal = [-(p + 2.0), -(q + 1.0), -4.0]
    A_ub_p = [
        [1.0, 1.0, 1.0],
        [-2.0, 1.0, -1.0]
    ]
    b_ub_p = [10.0 + k, -float(r)]
    A_eq_p = [[1.0, 2.0, -1.0]]
    b_eq_p = [4.0 + p]
    bounds_p = [(0.0, None), (0.0, None), (None, None)]
    
    res_primal = linprog(c_primal, A_ub=A_ub_p, b_ub=b_ub_p, A_eq=A_eq_p, b_eq=b_eq_p, 
                         bounds=bounds_p, method='highs')
    x_star = res_primal.x
    z_star = -res_primal.fun
    
    print("\nPrimal Solution:")
    print(f"  x1* = {x_star[0]:.4f}, x2* = {x_star[1]:.4f}, x3* = {x_star[2]:.4f}")
    print(f"  Primal optimum z* = {z_star:.4f}")
    
    # 2. Dual LP
    c_dual = [10.0 + k, float(r), 4.0 + p]
    A_ub_d = [
        [-1.0, -2.0, -1.0],
        [-1.0,  1.0, -2.0]
    ]
    b_ub_d = [-(p + 2.0), -(q + 1.0)]
    A_eq_d = [[1.0, 1.0, -1.0]]
    b_eq_d = [4.0]
    bounds_d = [(0.0, None), (None, 0.0), (None, None)]
    
    res_dual = linprog(c_dual, A_ub=A_ub_d, b_ub=b_ub_d, A_eq=A_eq_d, b_eq=b_eq_d,
                       bounds=bounds_d, method='highs')
    y_star = res_dual.x
    w_star = res_dual.fun
    
    print("\nDual Solution:")
    print(f"  y1* = {y_star[0]:.4f}, y2* = {y_star[1]:.4f}, y3* = {y_star[2]:.4f}")
    print(f"  Dual optimum w*   = {w_star:.4f}")
    
    # Strong Duality Check
    print(f"\nStrong Duality (z* == w*): {np.isclose(z_star, w_star, atol=1e-4)}")
    
    # Complementary Slackness Check
    s1 = (10.0 + k) - (x_star[0] + x_star[1] + x_star[2])
    e2 = (2.0 * x_star[0] - x_star[1] + x_star[2]) - r
    d1 = (y_star[0] + 2.0 * y_star[1] + y_star[2]) - (p + 2.0)
    d2 = (y_star[0] - y_star[1] + 2.0 * y_star[2]) - (q + 1.0)
    
    cs1 = y_star[0] * s1
    cs2 = y_star[1] * e2
    cs3 = x_star[0] * d1
    cs4 = x_star[1] * d2
    
    print("\nComplementary Slackness Products:")
    print(f"  y1* * s1 = {y_star[0]:.4f} * {s1:.4f} = {cs1:.4f}")
    print(f"  y2* * e2 = {y_star[1]:.4f} * {e2:.4f} = {cs2:.4f}")
    print(f"  x1* * d1 = {x_star[0]:.4f} * {d1:.4f} = {cs3:.4f}")
    print(f"  x2* * d2 = {x_star[1]:.4f} * {d2:.4f} = {cs4:.4f}")
    
    all_zero = (abs(cs1) < 1e-4 and abs(cs2) < 1e-4 and abs(cs3) < 1e-4 and abs(cs4) < 1e-4)
    print(f"Every complementary slackness product is zero: {all_zero}")

if __name__ == "__main__":
    run()
