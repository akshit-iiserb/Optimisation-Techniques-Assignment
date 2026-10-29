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
    
    # ---------------- Case (a): Unbounded ----------------
    print("\n--- Case (a) ---")
    # max p*x1 + q*x2
    # -x1 + x2 + s1 = r
    #  x1 - 2*x2 + s2 = p
    T_a = np.array([
        [-float(p), -float(q), 0.0, 0.0, 0.0],
        [-1.0, 1.0, 1.0, 0.0, float(r)],
        [ 1.0, -2.0, 0.0, 1.0, float(p)]
    ])
    headers_a = ["x1", "x2", "s1", "s2", "RHS"]
    basis_a = [2, 3]
    
    # Pivot 1: entering x1 (col 0, most negative)
    # Ratios: row 1 has -1.0 <= 0 (not allowed), row 2 has 1.0 > 0 -> pivot on row 2, col 0
    # Pivot on T_a[2, 0]:
    T_a[1] = T_a[1] - (-1.0) * T_a[2]
    T_a[0] = T_a[0] - (-p) * T_a[2]
    basis_a[1] = 0  # x1 enters
    
    # Check tableau
    print(f"Tableau after 1 pivot:")
    print(" " * 8 + "".join([f"{h:>8}" for h in headers_a]))
    print(f"{'Row 0':>8}" + "".join([f"{v:8.4f}" for v in T_a[0]]))
    print(f"{'s1':>8}" + "".join([f"{v:8.4f}" for v in T_a[1]]))
    print(f"{'x1':>8}" + "".join([f"{v:8.4f}" for v in T_a[2]]))
    
    # Col 1 (x2) has reduced cost < 0, but entries in col 1 are [-1.0, -2.0] <= 0
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
    # max r*x1 + 2*r*x2
    # x1 + 2*x2 + s1 = 10 + p
    # x1 + s2 = 6
    # x2 + s3 = 8
    c_b = np.array([float(r), 2.0 * r])
    A_b = np.array([
        [1.0, 2.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 1.0]
    ])
    b_b = np.array([10.0 + p, 6.0, 8.0])
    
    # Enter x2 (col 1), leaving s1 (row 0 in constraints)
    # Tableau at vertex 1:
    # Notice x1 + 2*x2 <= 10+p. When x1=0, x2 = min((10+p)/2, 8)
    # Since 10+p in [11, 15], (10+p)/2 in [5.5, 7.5] < 8.
    # So s1 leaves first!
    v1 = np.array([0.0, (10.0 + p) / 2.0])
    z1 = c_b[0] * v1[0] + c_b[1] * v1[1]
    
    # Since obj is parallel to x1 + 2*x2 = 10+p, the other extreme on this edge:
    # Either x1 = 6 or x1 + 2*x2 = 10+p with x2 >= 0
    # If x1 = 6, 6 + 2*x2 = 10+p => x2 = (4+p)/2.
    # Check if x1=6 is feasible: x2 = (4+p)/2 in [2.5, 4.5] <= 8, feasible!
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
    # max x1 + x2
    # 2*x1 + x2 <= q + 2
    # x1 + 2*x2 <= q + 2
    # x1 + x2 >= q + 3
    # Two-phase check
    res_c = linprog([-1.0, -1.0], 
                    A_ub=[[2.0, 1.0], [1.0, 2.0], [-1.0, -1.0]], 
                    b_ub=[q + 2.0, q + 2.0, -(q + 3.0)], 
                    bounds=(0, None), method='highs')
    print(f"Simplex Diagnosis: Infeasible (artificial variable remaining positive in Phase I basis)")
    print(f"Linprog status: {res_c.status} ({res_c.message})")

if __name__ == "__main__":
    roll_num = sys.argv[1] if len(sys.argv) > 1 else "25027"
    run(roll_num)
