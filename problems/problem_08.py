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

def print_tableau(T, headers, basis, iter_num):
    print(f"\n--- Tableau Iteration {iter_num} ---")
    row_fmt = "{:>8}" * len(headers)
    print(" " * 8 + row_fmt.format(*headers))
    print(f"{'Row 0':>8}" + "".join([f"{val:8.4f}" for val in T[0]]))
    for i in range(1, len(T)):
        b_var = headers[basis[i - 1]]
        print(f"{b_var:>8}" + "".join([f"{val:8.4f}" for val in T[i]]))

def run(roll="25027"):
    p, q, r, k = get_params(roll)
    print(f"Your roll number: {roll}   p = {p}   q = {q}   r = {r}   k = {k}")
    
    c = np.array([2.0 + p, 3.0 + q, 1.0 + r])
    b = np.array([10.0 + k, 12.0 + p, 18.0 + q])
    A = np.array([
        [1.0, 2.0, 1.0],
        [2.0, 1.0, 2.0],
        [1.0, 1.0, 3.0]
    ])
    
    headers = ["x1", "x2", "x3", "s1", "s2", "s3", "RHS"]
    
    T = np.zeros((4, 7))
    T[0, :3] = -c
    T[1:, :3] = A
    T[1:, 3:6] = np.eye(3)
    T[1:, 6] = b
    
    basis = [3, 4, 5]  # s1, s2, s3
    
    iter_count = 0
    while True:
        print_tableau(T, headers, basis, iter_count)
        
        row0 = T[0, :-1]
        if np.all(row0 >= -1e-7):
            print("\nOptimal tableau reached!")
            break
            
        entering = np.argmin(row0)
        
        ratios = []
        for i in range(1, 4):
            val = T[i, entering]
            if val > 1e-7:
                ratios.append(T[i, -1] / val)
            else:
                ratios.append(np.inf)
                
        ratios = np.array(ratios)
        if np.all(np.isinf(ratios)):
            print("\nProblem is unbounded!")
            return
            
        leaving = np.argmin(ratios) + 1
        pivot_val = T[leaving, entering]
        
        print(f"\nEntering variable: {headers[entering]}, Leaving variable: {headers[basis[leaving - 1]]}, Pivot: {pivot_val:.4f}")
        
        T[leaving] = T[leaving] / pivot_val
        for i in range(4):
            if i != leaving:
                T[i] = T[i] - T[i, entering] * T[leaving]
                
        basis[leaving - 1] = entering
        iter_count += 1
        
    sol = np.zeros(6)
    for i in range(1, 4):
        sol[basis[i - 1]] = T[i, -1]
        
    z_opt = T[0, -1]
    print(f"\nFinal Solution:")
    print(f"  x1* = {sol[0]:.4f}, x2* = {sol[1]:.4f}, x3* = {sol[2]:.4f}")
    print(f"  s1* = {sol[3]:.4f}, s2* = {sol[4]:.4f}, s3* = {sol[5]:.4f}")
    print(f"  Optimal z* = {z_opt:.4f}")
    
    res = linprog(-c, A_ub=A, b_ub=b, bounds=[(0, None)] * 3, method='highs')
    print(f"\nLinprog Cross-Check:")
    print(f"  x1* = {res.x[0]:.4f}, x2* = {res.x[1]:.4f}, x3* = {res.x[2]:.4f}")
    print(f"  Optimal z* = {-res.fun:.4f}")

if __name__ == "__main__":
    run()
