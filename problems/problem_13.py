import sys
import re
import numpy as np
from scipy.optimize import linprog

def get_params(roll):
    digits = re.sub(r"\D", "", str(roll))
    d = digits.zfill(4)[-4:]
    d1, d2, d3, d4 = map(int, d)
    p = d4 % 5 + 1
    q = d3 % 4 + 1
    r = d2 % 3 + 1
    k = (d1 + d2 + d3 + d4) % 9 + 1
    return p, q, r, k

def print_tableau(T, headers, basis, iter_num):
    print(f"\n--- Dual Simplex Tableau {iter_num} ---")
    row_fmt = "{:>8}" * len(headers)
    print(" " * 8 + row_fmt.format(*headers))
    print(f"{'Row 0':>8}" + "".join([f"{val:8.4f}" for val in T[0]]))
    for i in range(1, len(T)):
        b_var = headers[basis[i - 1]]
        print(f"{b_var:>8}" + "".join([f"{val:8.4f}" for val in T[i]]))

def run(roll="25027"):
    p, q, r, k = get_params(roll)
    print(f"Your roll number: {roll}   p = {p}   q = {q}   r = {r}   k = {k}")
    
    # min (p+1)*x1 + (q+1)*x2 + (r+1)*x3
    # s.t. -x1 - 2*x2 -   x3 + s1 = -(4 + k)
    #     -2*x1 -   x2 - 3*x3 + s2 = -(12 + p)
    #       -x1 -   x2 - 2*x3 + s3 = -(4 + q)
    c = np.array([float(p + 1), float(q + 1), float(r + 1), 0.0, 0.0, 0.0])
    headers = ["x1", "x2", "x3", "s1", "s2", "s3", "RHS"]
    
    T = np.zeros((4, 7))
    T[0, :6] = c
    T[1] = [-1.0, -2.0, -1.0, 1.0, 0.0, 0.0, -(4.0 + k)]
    T[2] = [-2.0, -1.0, -3.0, 0.0, 1.0, 0.0, -(12.0 + p)]
    T[3] = [-1.0, -1.0, -2.0, 0.0, 0.0, 1.0, -(4.0 + q)]
    
    basis = [3, 4, 5]  # s1, s2, s3
    
    iter_count = 0
    while True:
        print_tableau(T, headers, basis, iter_count)
        
        # Check primal feasibility (all RHS >= 0)
        rhs = T[1:, -1]
        if np.all(rhs >= -1e-7):
            print("\nOptimal and primal feasible tableau reached!")
            break
            
        # Leaving row: most negative RHS
        leaving = np.argmin(rhs) + 1  # 1-indexed row
        
        # Entering column: min ratio (c_j / |a_lj|) for a_lj < 0
        ratios = []
        for j in range(6):
            a_val = T[leaving, j]
            if a_val < -1e-7:
                ratios.append(T[0, j] / (-a_val))
            else:
                ratios.append(np.inf)
                
        ratios = np.array(ratios)
        if np.all(np.isinf(ratios)):
            print("\nDual simplex: Problem is primal infeasible!")
            return
            
        entering = np.argmin(ratios)
        pivot_val = T[leaving, entering]
        
        print(f"\nLeaving variable: {headers[basis[leaving - 1]]}, Entering variable: {headers[entering]}, Pivot: {pivot_val:.4f}")
        
        # Pivot operation
        T[leaving] = T[leaving] / pivot_val
        for i in range(4):
            if i != leaving:
                T[i] = T[i] - T[i, entering] * T[leaving]
                
        basis[leaving - 1] = entering
        iter_count += 1
        
    # Read solution
    sol = np.zeros(6)
    for i in range(1, 4):
        sol[basis[i - 1]] = T[i, -1]
        
    z_opt = -T[0, -1]
    print(f"\nFinal Optimal Solution:")
    print(f"  x1* = {sol[0]:.4f}, x2* = {sol[1]:.4f}, x3* = {sol[2]:.4f}")
    print(f"  s1* = {sol[3]:.4f}, s2* = {sol[4]:.4f}, s3* = {sol[5]:.4f}")
    print(f"  Optimal z* = {z_opt:.4f}")
    
    # Cross-check with linprog
    c_lp = [p + 1.0, q + 1.0, r + 1.0]
    A_ub = [
        [-1.0, -2.0, -1.0],
        [-2.0, -1.0, -3.0],
        [-1.0, -1.0, -2.0]
    ]
    b_ub = [-(4.0 + k), -(12.0 + p), -(4.0 + q)]
    res = linprog(c_lp, A_ub=A_ub, b_ub=b_ub, bounds=[(0, None)] * 3, method='highs')
    print(f"\nLinprog Cross-Check:")
    print(f"  x1* = {res.x[0]:.4f}, x2* = {res.x[1]:.4f}, x3* = {res.x[2]:.4f}")
    print(f"  Optimal z* = {res.fun:.4f}")

if __name__ == "__main__":
    roll_num = sys.argv[1] if len(sys.argv) > 1 else "25027"
    run(roll_num)
