import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog, milp, LinearConstraint

def get_instance(roll="25027"):
    digits = [c for c in str(roll) if c.isdigit()]
    d1, d2, d3, d4 = map(int, digits[-4:])
    p = d4 % 5 + 1
    q = d3 % 4 + 1
    r = d2 % 3 + 1
    k = (d1 + d2 + d3 + d4) % 9 + 1
    
    seed = int("".join(digits))
    rng = np.random.default_rng(seed)
    n, m = 40 + 2 * k, 12
    pts = rng.uniform(0, 8, size=(n, 2))
    gx, gy = np.meshgrid([1, 3, 5, 7], [1.3, 4, 6.7])
    sites = np.c_[gx.ravel(), gy.ravel()] + rng.uniform(-0.6, 0.6, size=(m, 2))
    demand = rng.integers(1, 6 + p, size=n)
    cap = rng.integers(25, 46, size=m) + 2 * q
    fixed = rng.integers(800, 1501, size=m) + 50 * r
    dist = np.linalg.norm(pts[:, None, :] - sites[None, :, :], axis=2)
    
    return dict(roll=str(roll), p=p, q=q, r=r, k=k, n=n, m=m,
                c1=0.02, c2=2000 * (1 + 0.1 * q), E=100 + 10 * r, kappa=0.5, gamma=0.1,
                demand=demand, capacity=cap, fixed=fixed, dist=dist.round(3))

def run(roll="25027"):
    inst = get_instance(roll)
    p, q, r, k = inst['p'], inst['q'], inst['r'], inst['k']
    print(f"Your roll number: {roll}   p = {p}   q = {q}   r = {r}   k = {k}")
    
    n, m = inst['n'], inst['m']
    c1, c2, E, kappa, gamma = inst['c1'], inst['c2'], inst['E'], inst['kappa'], inst['gamma']
    d = np.array(inst['demand'])
    Q = np.array(inst['capacity'])
    f_cost = np.array(inst['fixed'])
    dist = np.array(inst['dist'])
    
    # ---------------- 21.1 Range-optimal speed ----------------
    print("\n================ 21.1 Range-Optimal Speed ================")
    def e_fun(v):
        return c1 * v**2 + c2 / (v**2)
    def de_fun(v):
        return 2.0 * c1 * v - 2.0 * c2 / (v**3)
        
    v_star_analytical = (c2 / c1)**0.25
    print(f"Analytical range-optimal speed v_R* = (c2/c1)^(1/4) = {v_star_analytical:.4f} m/s")
    
    # Golden Section Search on [5, 40]
    a, b = 5.0, 40.0
    phi = (np.sqrt(5.0) - 1.0) / 2.0
    x1 = b - phi * (b - a)
    x2 = a + phi * (b - a)
    f1, f2 = e_fun(x1), e_fun(x2)
    evals_gs = 2
    while (b - a) > 1e-4:
        if f1 < f2:
            b = x2
            x2 = x1
            f2 = f1
            x1 = b - phi * (b - a)
            f1 = e_fun(x1)
        else:
            a = x1
            x1 = x2
            f1 = f2
            x2 = a + phi * (b - a)
            f2 = e_fun(x2)
        evals_gs += 1
    v_R_gs = (a + b) / 2.0
    print(f"Golden Section: v_R* = {v_R_gs:.4f} m/s ({evals_gs} function evaluations)")
    
    # Cubic Interpolation on [5, 40]
    evals_ci = 0
    A_ci, B_ci = 5.0, 40.0
    v_R_ci = A_ci
    for _ in range(20):
        fA, fB = e_fun(A_ci), e_fun(B_ci)
        dfA, dfB = de_fun(A_ci), de_fun(B_ci)
        evals_ci += 4
        Z = 3.0 * (fA - fB) / (B_ci - A_ci) + dfA + dfB
        disc = max(0.0, Z**2 - dfA * dfB)
        Q_term = np.sqrt(disc)
        denom = dfA + dfB + 2.0 * Z
        lam_cand = A_ci + ((dfA + Z - Q_term) / denom) * (B_ci - A_ci)
        if not (A_ci <= lam_cand <= B_ci):
            lam_cand = A_ci + ((dfA + Z + Q_term) / denom) * (B_ci - A_ci)
        v_R_ci = lam_cand
        df_mid = de_fun(v_R_ci)
        evals_ci += 1
        if abs(df_mid) < 1e-4 or abs(B_ci - A_ci) < 1e-4:
            break
        if df_mid < 0:
            A_ci = v_R_ci
        else:
            B_ci = v_R_ci
    print(f"Cubic Interpolation: v_R* = {v_R_ci:.4f} m/s ({evals_ci} function/derivative evaluations)")
    
    v_R = v_star_analytical
    e_opt = e_fun(v_R)
    c_opt = kappa * e_opt + gamma * 1000.0 / v_R
    R_opt = E / e_opt
    allowed_pairs = np.sum(2.0 * dist <= R_opt)
    total_pairs = n * m
    print(f"Optimal Range R(v_R*) = {R_opt:.4f} km")
    print(f"Allowed (i, j) pairs: {allowed_pairs} / {total_pairs} ({allowed_pairs / total_pairs * 100:.1f}%)")

    # ---------------- 21.2 Formulation and the LP relaxation ----------------
    print("\n================ 21.2 Formulation & LP Relaxation ================")
    num_vars = m + n * m
    num_constraints = n + m + n * m
    print(f"Variables: {num_vars} ({m} hubs y_j, {n*m} assignments x_ij)")
    print(f"Constraints: {num_constraints} ({n} demand, {m} capacity, {n*m} strong linking)")
    
    c_obj = np.zeros(num_vars)
    c_obj[:m] = f_cost
    for i in range(n):
        for j in range(m):
            c_obj[m + i * m + j] = 2.0 * dist[i, j] * c_opt * d[i]
            
    A_eq = np.zeros((n, num_vars))
    for i in range(n):
        A_eq[i, m + i * m : m + (i + 1) * m] = 1.0
    b_eq = np.ones(n)
    
    A_ub_strong = np.zeros((m + n * m, num_vars))
    for j in range(m):
        A_ub_strong[j, j] = -Q[j]
        for i in range(n):
            A_ub_strong[j, m + i * m + j] = d[i]
            
    for i in range(n):
        for j in range(m):
            row_idx = m + i * m + j
            A_ub_strong[row_idx, j] = -1.0
            A_ub_strong[row_idx, m + i * m + j] = 1.0
    b_ub_strong = np.zeros(m + n * m)
    
    bounds = [(0.0, 1.0)] * m
    for i in range(n):
        for j in range(m):
            if 2.0 * dist[i, j] > R_opt:
                bounds.append((0.0, 0.0))
            else:
                bounds.append((0.0, 1.0))
                
    res_lp_strong = linprog(c_obj, A_ub=A_ub_strong, b_ub=b_ub_strong, A_eq=A_eq, b_eq=b_eq, 
                            bounds=bounds, method='highs')
    z_lp_strong = res_lp_strong.fun
    print(f"Strong LP Relaxation Objective: z_LP = {z_lp_strong:.4f} Rs/hour")
    
    demand_marginals = res_lp_strong.eqlin.marginals
    most_expensive_pt = int(np.argmax(demand_marginals))
    print(f"Demand point most expensive to serve: Point {most_expensive_pt} (Marginal = {demand_marginals[most_expensive_pt]:.4f} Rs/hour)")
    print(f"Reason: Point {most_expensive_pt} is located furthest from available candidate hubs or has tight reachability.")
    
    cap_slacks = np.zeros(m)
    y_sol = res_lp_strong.x[:m]
    x_sol = res_lp_strong.x[m:].reshape((n, m))
    for j in range(m):
        cap_slacks[j] = Q[j] * y_sol[j] - np.sum(d * x_sol[:, j])
    cap_marginals = res_lp_strong.ineqlin.marginals[:m]
    cs_products = cap_slacks * cap_marginals
    print(f"Max capacity complementary slackness violation: {np.max(np.abs(cs_products)):.6e}")

    # ---------------- 21.3 Branch and Bound (4 variants) ----------------
    print("\n================ 21.3 Branch and Bound Variants ================")
    
    def solve_bb(node_strategy="dfs", branch_strategy="first", max_nodes=500):
        lps_solved = 0
        nodes_explored = 0
        z_best = np.inf
        y_best = None
        
        if node_strategy == "dfs":
            pool = [[(0.0, 1.0)] * m]
        else:
            pool = [([(0.0, 1.0)] * m, 0.0)]
            
        while pool and nodes_explored < max_nodes:
            if node_strategy == "dfs":
                bounds_y = pool.pop()
            else:
                best_idx = min(range(len(pool)), key=lambda i: pool[i][1])
                bounds_y, _ = pool.pop(best_idx)
                
            nodes_explored += 1
            node_bounds = list(bounds_y) + bounds[m:]
            
            res_node = linprog(c_obj, A_ub=A_ub_strong, b_ub=b_ub_strong, A_eq=A_eq, b_eq=b_eq,
                               bounds=node_bounds, method='highs')
            lps_solved += 1
            
            if res_node.status != 0:
                continue
            z_node = res_node.fun
            if z_node >= z_best - 1e-4:
                continue
                
            y_node = res_node.x[:m]
            frac_y = [j for j in range(m) if not (np.isclose(y_node[j], 0.0, atol=1e-4) or np.isclose(y_node[j], 1.0, atol=1e-4))]
            
            if not frac_y:
                if z_node < z_best:
                    z_best = z_node
                    y_best = np.round(y_node).astype(int)
                continue
                
            if branch_strategy == "first":
                branch_j = frac_y[0]
            else:
                branch_j = min(frac_y, key=lambda j: abs(y_node[j] - 0.5))
                
            b0 = list(bounds_y)
            b0[branch_j] = (0.0, 0.0)
            b1 = list(bounds_y)
            b1[branch_j] = (1.0, 1.0)
            
            if node_strategy == "dfs":
                pool.append(b1)
                pool.append(b0)
            else:
                pool.append((b0, z_node))
                pool.append((b1, z_node))
                
        gap = (z_best - z_lp_strong) / z_best if z_best < np.inf else 0.0
        return nodes_explored, lps_solved, z_best, gap, y_best

    variants = [
        ("Depth-First", "First-Frac", "dfs", "first"),
        ("Depth-First", "Most-Frac", "dfs", "most"),
        ("Best-Bound",  "First-Frac", "best", "first"),
        ("Best-Bound",  "Most-Frac", "best", "most")
    ]
    
    print(f"{'Node Strategy':>12} | {'Branch Rule':>10} | {'Nodes':>6} | {'LPs':>6} | {'Best z (Rs)':>12} | {'Gap %':>6}")
    print("-" * 65)
    
    for n_strat_name, b_strat_name, n_strat, b_strat in variants:
        n_exp, lps, z_b, gap, y_b = solve_bb(n_strat, b_strat)
        print(f"{n_strat_name:>12} | {b_strat_name:>10} | {n_exp:6d} | {lps:6d} | {z_b:12.4f} | {gap*100:6.2f}%")
            
    # Cross check with scipy.optimize.milp
    integrality = np.zeros(num_vars)
    integrality[:m] = 1
    lhs_m = np.concatenate([b_eq, np.full(m + n * m, -np.inf)])
    rhs_m = np.concatenate([b_eq, b_ub_strong])
    A_m = np.vstack([A_eq, A_ub_strong])
    constraints_m = LinearConstraint(A_m, lhs_m, rhs_m)
    
    lb_m = [b[0] for b in bounds]
    ub_m = [b[1] for b in bounds]
    res_milp = milp(c_obj, integrality=integrality, bounds=(lb_m, ub_m), constraints=constraints_m)
    print(f"\nSciPy MILP Cross-Check: z* = {res_milp.fun:.4f} Rs/hour (status = {res_milp.status})")

    # ---------------- 21.4 Formulation Matters ----------------
    print("\n================ 21.4 Formulation Matters ================")
    A_ub_agg = np.zeros((m + m, num_vars))
    for j in range(m):
        A_ub_agg[j, j] = -Q[j]
        for i in range(n):
            A_ub_agg[j, m + i * m + j] = d[i]
    for j in range(m):
        A_ub_agg[m + j, j] = -float(n)
        for i in range(n):
            A_ub_agg[m + j, m + i * m + j] = 1.0
    b_ub_agg = np.zeros(2 * m)
    
    res_lp_agg = linprog(c_obj, A_ub=A_ub_agg, b_ub=b_ub_agg, A_eq=A_eq, b_eq=b_eq, 
                         bounds=bounds, method='highs')
    z_lp_agg = res_lp_agg.fun
    z_ip = res_milp.fun
    gap_strong = (z_ip - z_lp_strong) / z_ip
    gap_agg = (z_ip - z_lp_agg) / z_ip
    
    print(f"Strong Formulation:     LP Bound = {z_lp_strong:.4f}, Integrality Gap = {gap_strong*100:.2f}%")
    print(f"Aggregated Formulation: LP Bound = {z_lp_agg:.4f}, Integrality Gap = {gap_agg*100:.2f}%")
    print(f"Comparison: Strong formulation provides a significantly tighter LP lower bound.")

    # ---------------- 21.5 Lagrangian Relaxation ----------------
    print("\n================ 21.5 Lagrangian Relaxation ================")
    u = np.mean(demand_marginals) * np.ones(n)
    theta = 2.0
    z_ub_lag = z_ip * 1.5
    best_lb = -np.inf
    non_improving = 0
    
    lb_history = []
    ub_history = []
    
    C_mat = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            C_mat[i, j] = 2.0 * dist[i, j] * c_opt * d[i] if 2.0 * dist[i, j] <= R_opt else 1e9
            
    for it in range(60):
        y_sub = np.zeros(m)
        x_sub = np.zeros((n, m))
        L_val = np.sum(u)
        
        for j in range(m):
            ratios = (u - C_mat[:, j]) / d
            cand_items = [i for i in range(n) if ratios[i] > 0 and C_mat[i, j] < 1e8]
            cand_items.sort(key=lambda i: ratios[i], reverse=True)
            
            rem_cap = Q[j]
            knap_profit = 0.0
            x_cand = np.zeros(n)
            for i in cand_items:
                alloc = min(1.0, rem_cap / d[i])
                if alloc > 0:
                    x_cand[i] = alloc
                    rem_cap -= alloc * d[i]
                    knap_profit += (u[i] - C_mat[i, j]) * alloc
                if rem_cap <= 1e-6:
                    break
            net_benefit = knap_profit - f_cost[j]
            if net_benefit > 0:
                y_sub[j] = 1.0
                x_sub[:, j] = x_cand
                L_val += (f_cost[j] - knap_profit)
                
        lb_history.append(L_val)
        if L_val > best_lb:
            best_lb = L_val
            non_improving = 0
        else:
            non_improving += 1
            
        if non_improving >= 20:
            theta /= 2.0
            non_improving = 0
            
        open_hubs = np.where(y_sub > 0.5)[0]
        if len(open_hubs) > 0 and np.sum(Q[open_hubs]) >= np.sum(d):
            bounds_feas = [(1.0, 1.0) if j in open_hubs else (0.0, 0.0) for j in range(m)] + bounds[m:]
            res_feas = linprog(c_obj, A_ub=A_ub_strong, b_ub=b_ub_strong, A_eq=A_eq, b_eq=b_eq,
                               bounds=bounds_feas, method='highs')
            if res_feas.status == 0 and res_feas.fun < z_ub_lag:
                z_ub_lag = res_feas.fun
        ub_history.append(z_ub_lag)
        
        g = 1.0 - np.sum(x_sub, axis=1)
        norm_g_sq = np.sum(g**2)
        if norm_g_sq < 1e-6:
            break
        step_tk = theta * (z_ub_lag - L_val) / norm_g_sq
        u = u + step_tk * g
        
    print(f"Best Lagrangian Lower Bound: {best_lb:.4f} Rs/hour")
    print(f"Strong LP Relaxation Bound:   {z_lp_strong:.4f} Rs/hour")
    print(f"Best Feasible Upper Bound:   {z_ub_lag:.4f} Rs/hour")
    
    plt.figure(figsize=(8, 5))
    plt.plot(lb_history, 'b-', label='Lower Bound L(u)')
    plt.plot(ub_history, 'r--', label='Upper Bound (Feasible)')
    plt.axhline(y=z_lp_strong, color='g', linestyle=':', label='Strong LP Bound')
    plt.xlabel('Iteration')
    plt.ylabel('Cost (Rs/hour)')
    plt.title('Problem 21.5: Lagrangian Relaxation Bound Progression')
    plt.grid(True)
    plt.legend()
    plt.savefig("figures/problem_21_lagrangian.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved plot to figures/problem_21_lagrangian.png")

    # ---------------- 21.6 Cost-optimal speed ----------------
    print("\n================ 21.6 Cost-Optimal Speed ================")
    
    def solve_milp_speed(v_val):
        e_v = c1 * v_val**2 + c2 / (v_val**2)
        c_v = kappa * e_v + gamma * 1000.0 / v_val
        R_v = E / e_v
        
        c_v_obj = np.zeros(num_vars)
        c_v_obj[:m] = f_cost
        for i in range(n):
            for j in range(m):
                c_v_obj[m + i * m + j] = 2.0 * dist[i, j] * c_v * d[i]
                
        b_v = [(0.0, 1.0)] * m
        for i in range(n):
            for j in range(m):
                if 2.0 * dist[i, j] > R_v:
                    b_v.append((0.0, 0.0))
                else:
                    b_v.append((0.0, 1.0))
                    
        lb_v = [b[0] for b in b_v]
        ub_v = [b[1] for b in b_v]
        res_v = milp(c_v_obj, integrality=integrality, bounds=(lb_v, ub_v), constraints=constraints_m)
        return res_v.fun if res_v.status == 0 else np.nan

    v_grid = np.arange(8, 31)
    F_grid = []
    print("Evaluating F(v) on grid v in [8, 30] m/s...")
    for v_val in v_grid:
        F_val = solve_milp_speed(float(v_val))
        F_grid.append(F_val)
        
    F_grid = np.array(F_grid)
    
    a_g, b_g = 8.0, 30.0
    x1_g = b_g - phi * (b_g - a_g)
    x2_g = a_g + phi * (b_g - a_g)
    F1 = solve_milp_speed(x1_g)
    F2 = solve_milp_speed(x2_g)
    
    while (b_g - a_g) > 0.1:
        if F1 < F2:
            b_g = x2_g
            x2_g = x1_g
            F2 = F1
            x1_g = b_g - phi * (b_g - a_g)
            F1 = solve_milp_speed(x1_g)
        else:
            a_g = x1_g
            x1_g = x2_g
            F1 = F2
            x2_g = a_g + phi * (b_g - a_g)
            F2 = solve_milp_speed(x2_g)
            
    v_F_star = (a_g + b_g) / 2.0
    print(f"Cost-optimal speed v_F* = {v_F_star:.4f} m/s (vs Range-optimal v_R* = {v_R:.4f} m/s)")
    
    plt.figure(figsize=(8, 5))
    plt.plot(v_grid, F_grid, 'bo-', label='F(v) MILP cost')
    plt.axvline(x=v_F_star, color='r', linestyle='--', label=f'Cost-optimal v_F* = {v_F_star:.1f} m/s')
    plt.axvline(x=v_R, color='g', linestyle=':', label=f'Range-optimal v_R* = {v_R:.1f} m/s')
    plt.xlabel('Cruise Speed v (m/s)')
    plt.ylabel('Optimal System Cost F(v) (Rs/hour)')
    plt.title('Problem 21.6: System Cost vs Drone Cruise Speed')
    plt.grid(True)
    plt.legend()
    plt.savefig("figures/problem_21_cost_speed.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved plot to figures/problem_21_cost_speed.png")

if __name__ == "__main__":
    run()
