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

def solve_simplex_tableau(p, q, r, k):
    c = np.array([2.0 + p, 3.0 + q, 1.0 + r])
    b = np.array([10.0 + k, 12.0 + p, 18.0 + q])
    A = np.array([
        [1.0, 2.0, 1.0],
        [2.0, 1.0, 2.0],
        [1.0, 1.0, 3.0]
    ])
    T = np.zeros((4, 7))
    T[0, :3] = -c
    T[1:, :3] = A
    T[1:, 3:6] = np.eye(3)
    T[1:, 6] = b
    basis = [3, 4, 5]
    
    while True:
        row0 = T[0, :-1]
        if np.all(row0 >= -1e-7):
            break
        entering = np.argmin(row0)
        ratios = []
        for i in range(1, 4):
            val = T[i, entering]
            ratios.append(T[i, -1] / val if val > 1e-7 else np.inf)
        leaving = np.argmin(ratios) + 1
        pivot_val = T[leaving, entering]
        T[leaving] /= pivot_val
        for i in range(4):
            if i != leaving:
                T[i] -= T[i, entering] * T[leaving]
        basis[leaving - 1] = entering
        
    y_star = T[0, 3:6]
    z_star = T[0, -1]
    return y_star, z_star

def run(roll="25027"):
    p, q, r, k = get_params(roll)
    print(f"Your roll number: {roll}   p = {p}   q = {q}   r = {r}   k = {k}")
    
    y_star_tableau, z_star = solve_simplex_tableau(p, q, r, k)
    print("\nOptimal Dual Values from Problem 8 Final Tableau:")
    print(f"  y1* = {y_star_tableau[0]:.4f}, y2* = {y_star_tableau[1]:.4f}, y3* = {y_star_tableau[2]:.4f}")
    
    c_dual = [10.0 + k, 12.0 + p, 18.0 + q]
    A_ub_d = [
        [-1.0, -2.0, -1.0],
        [-2.0, -1.0, -1.0],
        [-1.0, -2.0, -3.0]
    ]
    b_ub_d = [-(2.0 + p), -(3.0 + q), -(1.0 + r)]
    bounds_d = [(0.0, None)] * 3
    
    res_dual = linprog(c_dual, A_ub=A_ub_d, b_ub=b_ub_d, bounds=bounds_d, method='highs')
    print(f"\nLinprog Dual Solution:")
    print(f"  y1* = {res_dual.x[0]:.4f}, y2* = {res_dual.x[1]:.4f}, y3* = {res_dual.x[2]:.4f}")
    print(f"  Dual values match tableau: {np.allclose(y_star_tableau, res_dual.x, atol=1e-4)}")
    
    c_primal = [-(2.0 + p), -(3.0 + q), -(1.0 + r)]
    A_ub_p = [
        [1.0, 2.0, 1.0],
        [2.0, 1.0, 2.0],
        [1.0, 1.0, 3.0]
    ]
    b_ub_p_orig = [10.0 + k, 12.0 + p, 18.0 + q]
    b_ub_p_new  = [10.0 + k + 1.0, 12.0 + p, 18.0 + q]
    
    res_p_orig = linprog(c_primal, A_ub=A_ub_p, b_ub=b_ub_p_orig, bounds=[(0.0, None)] * 3, method='highs')
    res_p_new  = linprog(c_primal, A_ub=A_ub_p, b_ub=b_ub_p_new, bounds=[(0.0, None)] * 3, method='highs')
    
    delta_z = (-res_p_new.fun) - (-res_p_orig.fun)
    print(f"\nPerturbation check (b1 -> b1 + 1):")
    print(f"  Original z* = {-res_p_orig.fun:.4f}")
    print(f"  New z*      = {-res_p_new.fun:.4f}")
    print(f"  Delta z*    = {delta_z:.4f}")
    print(f"  y1*         = {y_star_tableau[0]:.4f}")
    print(f"  Delta z* == y1*: {np.isclose(delta_z, y_star_tableau[0], atol=1e-4)}")
    
    print("\nEconomic Explanation:")
    print("A non-binding resource has excess unused capacity at optimality (positive slack).")
    print("Providing more of an unused resource cannot improve the objective, so its marginal value (shadow price) is zero.")

if __name__ == "__main__":
    run()
