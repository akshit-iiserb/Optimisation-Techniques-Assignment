import sys
import re
import numpy as np
from scipy.optimize import minimize

def get_params(roll):
    digits = re.sub(r"\D", "", str(roll))
    d = digits.zfill(4)[-4:]
    d1, d2, d3, d4 = map(int, d)
    p = d4 % 5 + 1
    q = d3 % 4 + 1
    r = d2 % 3 + 1
    k = (d1 + d2 + d3 + d4) % 9 + 1
    return p, q, r, k

def solve_numerical(v_val, x_init=(3.0, 6.0)):
    def obj(x):
        return 2.0 * np.pi * x[0]**2 + 2.0 * np.pi * x[0] * x[1]
    cons = ({'type': 'eq', 'fun': lambda x: np.pi * (x[0]**2) * x[1] - v_val})
    bounds = [(1e-5, None), (1e-5, None)]
    res = minimize(obj, x_init, method='SLSQP', bounds=bounds, constraints=cons, tol=1e-9)
    return res.x[0], res.x[1], res.fun

def run(roll="25027"):
    p, q, r, k = get_params(roll)
    print(f"Your roll number: {roll}   p = {p}   q = {q}   r = {r}   k = {k}")
    
    V0 = 50.0 * (p + q)
    
    # Analytical solution
    x1_star = (V0 / (2.0 * np.pi))**(1.0 / 3.0)
    x2_star = 2.0 * x1_star
    S_star = 6.0 * np.pi * (x1_star**2)
    lam_star = 2.0 / x1_star
    delta_S_pred = lam_star * 1.0
    
    print(f"\nAnalytical solution at V0 = {V0:.4f}:")
    print(f"  x1* (radius) = {x1_star:.4f}")
    print(f"  x2* (height) = {x2_star:.4f}")
    print(f"  S* (area)    = {S_star:.4f}")
    print(f"  lambda*      = {lam_star:.4f}")
    print(f"  Predicted delta S* for delta V0 = 1: {delta_S_pred:.4f}")
    
    # Numerical solutions
    x1_num0, x2_num0, S_num0 = solve_numerical(V0)
    x1_num1, x2_num1, S_num1 = solve_numerical(V0 + 1.0, (x1_num0, x2_num0))
    delta_S_actual = S_num1 - S_num0
    
    print(f"\nNumerical solution at V0 = {V0:.4f}:")
    print(f"  x1 = {x1_num0:.4f}, x2 = {x2_num0:.4f}, S = {S_num0:.4f}")
    
    print(f"Numerical solution at V0 + 1 = {V0 + 1.0:.4f}:")
    print(f"  x1 = {x1_num1:.4f}, x2 = {x2_num1:.4f}, S = {S_num1:.4f}")
    
    print(f"\nActual change in S*:    {delta_S_actual:.4f}")
    print(f"Predicted change in S*: {delta_S_pred:.4f}")
    print(f"Absolute difference:    {abs(delta_S_actual - delta_S_pred):.4f}")

if __name__ == "__main__":
    roll_num = sys.argv[1] if len(sys.argv) > 1 else "25027"
    run(roll_num)
