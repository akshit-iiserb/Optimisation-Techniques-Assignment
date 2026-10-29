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

def print_tableau(T, headers, basis, label):
    print(f"\n--- {label} ---")
    row_fmt = "{:>8}" * len(headers)
    print(" " * 8 + row_fmt.format(*headers))
    print(f"{'Row 0':>8}" + "".join([f"{val:8.4f}" for val in T[0]]))
    for i in range(1, len(T)):
        b_var = headers[basis[i - 1]]
        print(f"{b_var:>8}" + "".join([f"{val:8.4f}" for val in T[i]]))

def pivot(T, basis, entering, leaving):
    pivot_val = T[leaving, entering]
    T[leaving] = T[leaving] / pivot_val
    for i in range(len(T)):
        if i != leaving:
            T[i] = T[i] - T[i, entering] * T[leaving]
    basis[leaving - 1] = entering

def run_simplex(T, headers, basis, max_iter=20):
    for _ in range(max_iter):
        row0 = T[0, :-1]
        if np.all(row0 >= -1e-7):
            break
        entering = np.argmin(row0)
        ratios = []
        for i in range(1, len(T)):
            val = T[i, entering]
            if val > 1e-7:
                ratios.append(T[i, -1] / val)
            else:
                ratios.append(np.inf)
        ratios = np.array(ratios)
        if np.all(np.isinf(ratios)):
            print("Unbounded in simplex phase!")
            return False
        leaving = np.argmin(ratios) + 1
        pivot(T, basis, entering, leaving)
    return True

def run(roll="25027"):
    p, q, r, k = get_params(roll)
    print(f"Your roll number: {roll}   p = {p}   q = {q}   r = {r}   k = {k}")
    
    headers_phase1 = ["x1", "x2", "e1", "a1", "e2", "a2", "s3", "RHS"]
    T1 = np.zeros((4, 8))
    
    T1[1] = [1.0, 1.0, -1.0, 1.0,  0.0, 0.0, 0.0, 4.0 + r]
    T1[2] = [1.0, 3.0,  0.0, 0.0, -1.0, 1.0, 0.0, 6.0 + q]
    T1[3] = [1.0, 1.0,  0.0, 0.0,  0.0, 0.0, 1.0, 12.0 + k]
    
    T1[0] = -(T1[1] + T1[2])
    T1[0, 3] = 0.0
    T1[0, 5] = 0.0
    
    basis1 = [3, 5, 6]  # a1, a2, s3
    
    print_tableau(T1, headers_phase1, basis1, "Phase I Initial Tableau")
    run_simplex(T1, headers_phase1, basis1)
    print_tableau(T1, headers_phase1, basis1, "End of Phase I Tableau")
    
    w_val = -T1[0, -1]
    print(f"Phase I optimal value W* = {abs(w_val):.4f}")
    if abs(w_val) > 1e-4:
        print("Problem is infeasible!")
        return
        
    keep_indices = [0, 1, 2, 4, 6, 7]  # x1, x2, e1, e2, s3, RHS
    headers_phase2 = ["x1", "x2", "e1", "e2", "s3", "RHS"]
    
    T2 = np.zeros((4, 6))
    T2[1:] = T1[1:, keep_indices]
    
    idx_map = {0: 0, 1: 1, 2: 2, 4: 3, 6: 4}
    basis2 = [idx_map[b] for b in basis1]
    
    c_p2 = [p + 1.0, q + 2.0, 0.0, 0.0, 0.0, 0.0]
    T2[0] = c_p2
    
    for i in range(1, 4):
        b_var = basis2[i - 1]
        T2[0] = T2[0] - T2[0, b_var] * T2[i]
        
    run_simplex(T2, headers_phase2, basis2)
    print_tableau(T2, headers_phase2, basis2, "Phase II Optimum Tableau")
    
    sol2 = np.zeros(5)
    for i in range(1, 4):
        sol2[basis2[i - 1]] = T2[i, -1]
        
    z_opt = -T2[0, -1]
    print(f"\nPhase II Optimum Solution:")
    print(f"  x1* = {sol2[0]:.4f}, x2* = {sol2[1]:.4f}")
    print(f"  e1* = {sol2[2]:.4f}, e2* = {sol2[3]:.4f}, s3* = {sol2[4]:.4f}")
    print(f"  Optimal z* = {z_opt:.4f}")
    
    c_lp = [p + 1.0, q + 2.0]
    A_ub = [
        [-1.0, -1.0],
        [-1.0, -3.0],
        [1.0, 1.0]
    ]
    b_ub = [-(4.0 + r), -(6.0 + q), 12.0 + k]
    res = linprog(c_lp, A_ub=A_ub, b_ub=b_ub, bounds=[(0, None), (0, None)], method='highs')
    print(f"\nLinprog Cross-Check:")
    print(f"  x1* = {res.x[0]:.4f}, x2* = {res.x[1]:.4f}, z* = {res.fun:.4f}")

if __name__ == "__main__":
    run()
