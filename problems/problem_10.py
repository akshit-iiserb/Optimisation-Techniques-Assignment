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
    
    # ---------------- Case (a): Unbounded ----------------
    print("\n--- Case (a) ---")
    T_a = np.array([
        [-float(p), -float(q), 0.0, 0.0, 0.0],
        [-1.0, 1.0, 1.0, 0.0, float(r)],
        [ 1.0, -2.0, 0.0, 1.0, float(p)]
    ])
    headers_a = ["x1", "x2", "s1", "s2", "RHS"]
    
    # Pivot on T_a[2, 0]
    T_a[1] = T_a[1] - (-1.0) * T_a[2]
    T_a[0] = T_a[0] - (-p) * T_a[2]
    
    print(f"Tableau after 1 pivot:")
    print(" " * 8 + "".join([f"{h:>8}" for h in headers_a]))
    print(f"{'Row 0':>8}" + "".join([f"{v:8.4f}" for v in T_a[0]]))
    print(f"{'s1':>8}" + "".join([f"{v:8.4f}" for v in T_a[1]]))
    print(f"{'x1':>8}" + "".join([f"{v:8.4f}" for v in T_a[2]]))
    
    col_x2 = T_a[1:, 1]
    if T_a[0, 1] < -1e-7 and np.all(col_x2 <= 1e-7):
        diag_a = "Unbounded (entering column has negative reduced cost and no positive entry)"
    else:
        diag_a = "Other"
        
    res_a = linprog([-p, -q], A_ub=[[-1.0, 1.0], [1.0, -2.0]], b_ub=[r, p], bounds=(0, None), method='highs')
    print(f"Simplex Diagnosis: {diag_a}")
    print(f"Linprog status: {res_a.status} ({res_a.message})")

    # ---------------- Case (b): Alternative Optima ----------------
    print("\n--- Case (b) ---")
    c_b = np.array([float(r), 2.0 * r])
    
    v1 = np.array([0.0, (10.0 + p) / 2.0])
    z1 = c_b[0] * v1[0] + c_b[1] * v1[1]
    
    v2 = np.array([min(6.0, 10.0 + p), max(0.0, (10.0 + p - 6.0) / 2.0)])
    z2 = c_b[0] * v2[0] + c_b[1] * v2[1]
    
    print(f"Optimal Vertex 1: ({v1[0]:.4f}, {v1[1]:.4f}) with z = {z1:.4f}")
    print(f"Optimal Vertex 2: ({v2[0]:.4f}, {v2[1]:.4f}) with z = {z2:.4f}")
    print(f"Parametric Form of All Optima:")
    print(f"  x(theta) = theta * [{v1[0]:.4f}, {v1[1]:.4f}] + (1 - theta) * [{v2[0]:.4f}, {v2[1]:.4f}]  for theta in [0, 1]")
    
    res_b = linprog([-c_b[0], -c_b[1]], A_ub=[[1.0, 2.0], [1.0, 0.0], [0.0, 1.0]], 
                    b_ub=[10.0 + p, 6.0, 8.0], bounds=(0, None), method='highs')
    print(f"Simplex Diagnosis: Alternative Optima (zero reduced cost for non-basic variable at optimal vertex)")
    print(f"Linprog status: {res_b.status} ({res_b.message})")

    # ---------------- Case (c): Infeasible ----------------
    print("\n--- Case (c) ---")
    res_c = linprog([-1.0, -1.0], 
                    A_ub=[[2.0, 1.0], [1.0, 2.0], [-1.0, -1.0]], 
                    b_ub=[q + 2.0, q + 2.0, -(q + 3.0)], 
                    bounds=(0, None), method='highs')
    print(f"Simplex Diagnosis: Infeasible (artificial variable remaining positive in Phase I basis)")
    print(f"Linprog status: {res_c.status} ({res_c.message})")

if __name__ == "__main__":
    run()
