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

def run(roll="25027"):
    p, q, r, k = get_params(roll)
    print(f"Your roll number: {roll}   p = {p}   q = {q}   r = {r}   k = {k}")
    
    # LP relaxation data
    c = np.array([-(3.0 + p), -(2.0 + q)])
    A_ub = np.array([
        [5.0, 4.0],
        [2.0, 5.0]
    ])
    b_ub = np.array([25.0 + 3.0 * r, 12.0 + p])
    
    print("\n--- Depth-First Branch and Bound ---")
    
    # Stack for DFS: list of nodes
    # Node structure: (node_id, parent_id, branch_desc, bounds)
    stack = [(1, 0, "Root", [(0.0, None), (0.0, None)])]
    
    node_counter = 1
    z_best = -np.inf
    x_best = None
    
    print(f"{'Node':>4} | {'Parent':>6} | {'Branch':>12} | {'x1_LP':>7} | {'x2_LP':>7} | {'z_LP':>7} | Status")
    print("-" * 75)
    
    while stack:
        node_id, parent_id, branch_desc, bounds = stack.pop()
        
        # Solve LP relaxation
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        if res.status != 0:
            status = "Infeasible"
            print(f"{node_id:4d} | {parent_id:6d} | {branch_desc:>12} | {'-':>7} | {'-':>7} | {'-':>7} | {status}")
            continue
            
        z_lp = -res.fun
        x_lp = res.x
        
        # Bound pruning
        if z_lp <= z_best + 1e-6:
            status = f"Fathomed by bound (z_lp <= {z_best:.2f})"
            print(f"{node_id:4d} | {parent_id:6d} | {branch_desc:>12} | {x_lp[0]:7.4f} | {x_lp[1]:7.4f} | {z_lp:7.4f} | {status}")
            continue
            
        # Integrality check
        is_int_0 = np.isclose(x_lp[0], np.round(x_lp[0]), atol=1e-5)
        is_int_1 = np.isclose(x_lp[1], np.round(x_lp[1]), atol=1e-5)
        
        if is_int_0 and is_int_1:
            status = "Integer (new incumbent)"
            z_best = z_lp
            x_best = np.round(x_lp).astype(int)
            print(f"{node_id:4d} | {parent_id:6d} | {branch_desc:>12} | {x_lp[0]:7.4f} | {x_lp[1]:7.4f} | {z_lp:7.4f} | {status}")
            continue
            
        # Branch on first fractional variable
        frac_idx = 0 if not is_int_0 else 1
        val = np.floor(x_lp[frac_idx])
        
        status = f"Branched on x{frac_idx+1}"
        print(f"{node_id:4d} | {parent_id:6d} | {branch_desc:>12} | {x_lp[0]:7.4f} | {x_lp[1]:7.4f} | {z_lp:7.4f} | {status}")
        
        # Child 2: >= ceil(val)
        bounds_ge = [list(bounds[0]), list(bounds[1])]
        bounds_ge[frac_idx][0] = max(bounds_ge[frac_idx][0] or 0.0, val + 1.0)
        bounds_ge = [(b[0], b[1]) for b in bounds_ge]
        
        # Child 1: <= floor(val)
        bounds_le = [list(bounds[0]), list(bounds[1])]
        bounds_le[frac_idx][1] = min(bounds_le[frac_idx][1] or np.inf, val)
        bounds_le = [(b[0], b[1]) for b in bounds_le]
        
        node_counter += 1
        node_ge_id = node_counter
        node_counter += 1
        node_le_id = node_counter
        
        # Push >= child first, then <= child so that <= child is popped and processed first
        stack.append((node_ge_id, node_id, f"x{frac_idx+1} >= {val+1:.0f}", bounds_ge))
        stack.append((node_le_id, node_id, f"x{frac_idx+1} <= {val:.0f}", bounds_le))
        
    print(f"\nBranch and Bound Optimum:")
    print(f"  x1* = {x_best[0]}, x2* = {x_best[1]}")
    print(f"  Optimal z* = {z_best:.4f}")
    
    # Brute-force enumeration of integer points
    max_x1 = int(np.floor((25.0 + 3.0 * r) / 5.0))
    max_x2 = int(np.floor((12.0 + p) / 5.0))
    
    bf_best_z = -np.inf
    bf_best_x = None
    
    for i in range(max_x1 + 1):
        for j in range(max_x2 + 1):
            if 5.0 * i + 4.0 * j <= 25.0 + 3.0 * r and 2.0 * i + 5.0 * j <= 12.0 + p:
                z_cand = (3.0 + p) * i + (2.0 + q) * j
                if z_cand > bf_best_z:
                    bf_best_z = z_cand
                    bf_best_x = (i, j)
                    
    print(f"\nBrute-force Enumeration Optimum:")
    print(f"  x1* = {bf_best_x[0]}, x2* = {bf_best_x[1]}")
    print(f"  Optimal z* = {bf_best_z:.4f}")
    print(f"Confirmation match: {np.isclose(z_best, bf_best_z, atol=1e-4) and (x_best[0] == bf_best_x[0]) and (x_best[1] == bf_best_x[1])}")

if __name__ == "__main__":
    roll_num = sys.argv[1] if len(sys.argv) > 1 else "25027"
    run(roll_num)
