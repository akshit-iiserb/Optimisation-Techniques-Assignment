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
    
    # Part (a): Original vs Standard Form
    print("\n--- Part (a) Original vs Standard Form ---")
    c_orig = [float(p), -float(q), 3.0]
    A_ub_orig = [
        [-1.0, -1.0, 1.0],
        [2.0, -1.0, 1.0]
    ]
    b_ub_orig = [-float(r), 20.0 + k]
    A_eq_orig = [[1.0, 1.0, 1.0]]
    b_eq_orig = [6.0 + p]
    bounds_orig = [(0.0, None), (None, 0.0), (None, None)]
    
    res_orig = linprog(c_orig, A_ub=A_ub_orig, b_ub=b_ub_orig, A_eq=A_eq_orig, b_eq=b_eq_orig, 
                       bounds=bounds_orig, method='highs')
    print(f"Original LP solution:      z* = {res_orig.fun:.4f}")
    print(f"  x1 = {res_orig.x[0]:.4f}, x2 = {res_orig.x[1]:.4f}, x3 = {res_orig.x[2]:.4f}")
    
    c_std = [float(p), float(q), 3.0, -3.0, 0.0, 0.0]
    A_eq_std = [
        [1.0, -1.0, -1.0, 1.0, -1.0, 0.0],
        [2.0, 1.0, 1.0, -1.0, 0.0, 1.0],
        [1.0, -1.0, 1.0, -1.0, 0.0, 0.0]
    ]
    b_eq_std = [float(r), 20.0 + k, 6.0 + p]
    bounds_std = [(0.0, None)] * 6
    
    res_std = linprog(c_std, A_eq=A_eq_std, b_eq=b_eq_std, bounds=bounds_std, method='highs')
    print(f"Standard Form LP solution: z* = {res_std.fun:.4f}")
    print(f"  Optimal z matches: {np.isclose(res_orig.fun, res_std.fun, atol=1e-4)}")
    
    # Part (b): Enumerate all 6 basic solutions
    print("\n--- Part (b) Basic Solutions Enumeration ---")
    A = np.array([
        [2.0, 1.0, 1.0, 0.0],
        [1.0, 3.0, 0.0, 1.0]
    ])
    b = np.array([8.0 + r, 9.0 + k])
    var_names = ["x1", "x2", "s1", "s2"]
    c_b = [(p + 1.0), (q + 1.0), 0.0, 0.0]
    
    best_bfs_z = -1e9
    best_bfs = None
    sol_idx = 1
    
    for i in range(4):
        for j in range(i + 1, 4):
            basic_cols = [i, j]
            B = A[:, basic_cols]
            if abs(np.linalg.det(B)) > 1e-9:
                xB = np.linalg.solve(B, b)
                sol = np.zeros(4)
                sol[basic_cols] = xB
                feasible = np.all(xB >= -1e-7)
                z_val = c_b[0] * sol[0] + c_b[1] * sol[1]
                status = "Feasible (BFS)" if feasible else "Infeasible"
                
                b_vars = f"({var_names[i]}, {var_names[j]})"
                print(f"Sol {sol_idx}: Basic {b_vars:10s} -> [x1, x2, s1, s2] = [{sol[0]:.2f}, {sol[1]:.2f}, {sol[2]:.2f}, {sol[3]:.2f}] | z = {z_val:.4f} | {status}")
                
                if feasible and z_val > best_bfs_z:
                    best_bfs_z = z_val
                    best_bfs = (b_vars, sol)
                sol_idx += 1
            
    print(f"\nBest Basic Feasible Solution: Basic variables {best_bfs[0]} with z* = {best_bfs_z:.4f}")

if __name__ == "__main__":
    run()
