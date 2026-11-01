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
    
    v = np.array([10.0 + p, 13.0 + q, 7.0 + r, 8.0 + k, 11.0, 9.0])
    w = np.array([4.0, 6.0, 3.0, 5.0, 5.0, 4.0])
    W = 13.0 + r
    
    print(f"\nKnapsack Capacity W = {W:.0f}")
    print(f"Values  v = {v.tolist()}")
    print(f"Weights w = {w.tolist()}")
    
    c = -v
    A_ub = [w]
    b_ub = [W]
    
    init_bounds = [(0.0, 1.0)] * 6
    stack = [(1, 0, "Root", init_bounds)]
    
    node_counter = 1
    nodes_explored = 0
    z_best = -np.inf
    x_best = None
    
    print("\n--- Branch and Bound Tree Exploration ---")
    print(f"{'Node':>4} | {'Parent':>6} | {'Branch':>10} | {'z_LP':>7} | Status")
    print("-" * 55)
    
    while stack:
        node_id, parent_id, branch_desc, bounds = stack.pop()
        nodes_explored += 1
        
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        if res.status != 0:
            status = "Infeasible"
            print(f"{node_id:4d} | {parent_id:6d} | {branch_desc:>10} | {'-':>7} | {status}")
            continue
            
        z_lp = -res.fun
        x_lp = res.x
        
        if z_lp <= z_best + 1e-6:
            status = f"Fathomed by bound (<= {z_best:.1f})"
            print(f"{node_id:4d} | {parent_id:6d} | {branch_desc:>10} | {z_lp:7.4f} | {status}")
            continue
            
        frac_indices = [i for i in range(6) if not np.isclose(x_lp[i], np.round(x_lp[i]), atol=1e-5)]
        
        if len(frac_indices) == 0:
            status = "Integer (new incumbent)"
            z_best = z_lp
            x_best = np.round(x_lp).astype(int)
            print(f"{node_id:4d} | {parent_id:6d} | {branch_desc:>10} | {z_lp:7.4f} | {status}")
            continue
            
        frac_idx = frac_indices[0]
        status = f"Branched on x{frac_idx+1}"
        print(f"{node_id:4d} | {parent_id:6d} | {branch_desc:>10} | {z_lp:7.4f} | {status}")
        
        bounds_ge = [list(b) for b in bounds]
        bounds_ge[frac_idx] = (1.0, 1.0)
        bounds_ge = [(b[0], b[1]) for b in bounds_ge]
        
        bounds_le = [list(b) for b in bounds]
        bounds_le[frac_idx] = (0.0, 0.0)
        bounds_le = [(b[0], b[1]) for b in bounds_le]
        
        node_counter += 1
        node_ge_id = node_counter
        node_counter += 1
        node_le_id = node_counter
        
        stack.append((node_ge_id, node_id, f"x{frac_idx+1} = 1", bounds_ge))
        stack.append((node_le_id, node_id, f"x{frac_idx+1} = 0", bounds_le))
        
    print(f"\nBranch and Bound Results:")
    print(f"  Optimal x* = {x_best.tolist()}")
    print(f"  Optimal Value z* = {z_best:.4f}")
    print(f"  Total Nodes Explored = {nodes_explored}")
    
    # Beginner-friendly 6-loop brute force
    bf_best_z = -np.inf
    bf_best_x = None
    
    for x0 in [0, 1]:
      for x1 in [0, 1]:
        for x2 in [0, 1]:
          for x3 in [0, 1]:
            for x4 in [0, 1]:
              for x5 in [0, 1]:
                combo_arr = np.array([x0, x1, x2, x3, x4, x5])
                weight_tot = np.dot(w, combo_arr)
                if weight_tot <= W:
                    val_tot = np.dot(v, combo_arr)
                    if val_tot > bf_best_z:
                        bf_best_z = val_tot
                        bf_best_x = combo_arr
                
    print(f"\nBrute-Force Enumeration (64 combinations):")
    print(f"  Optimal x* = {bf_best_x.tolist()}")
    print(f"  Optimal Value z* = {bf_best_z:.4f}")
    match = np.isclose(z_best, bf_best_z, atol=1e-4)
    print(f"Optimal objective value confirmed: {match}")

if __name__ == "__main__":
    run()
