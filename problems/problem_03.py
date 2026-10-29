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

def run(roll="25027"):
    p, q, r, k = get_params(roll)
    print(f"Your roll number: {roll}   p = {p}   q = {q}   r = {r}   k = {k}")
    
    C = 10 + k
    
    # 1. Lagrange Multipliers Analytical
    lam_star = 2.0 * C / (1.0 / p + 1.0 / q + 1.0 / r)
    x1_lag = lam_star / (2.0 * p)
    x2_lag = lam_star / (2.0 * q)
    x3_lag = lam_star / (2.0 * r)
    f_lag = p * x1_lag**2 + q * x2_lag**2 + r * x3_lag**2
    
    print("\n1. Lagrange Multiplier Analytical Solution:")
    print(f"  x1* = {x1_lag:.4f}, x2* = {x2_lag:.4f}, x3* = {x3_lag:.4f}")
    print(f"  lambda* = {lam_star:.4f}, f* = {f_lag:.4f}")
    
    # 2. Direct Substitution (eliminate x3 = C - x1 - x2)
    def f_sub(x):
        x1, x2 = x[0], x[1]
        x3 = C - x1 - x2
        return p * x1**2 + q * x2**2 + r * x3**2
        
    res_sub = minimize(f_sub, [1.0, 1.0], method='BFGS')
    x1_sub = res_sub.x[0]
    x2_sub = res_sub.x[1]
    x3_sub = C - x1_sub - x2_sub
    f_sub_val = res_sub.fun
    
    print("\n2. Direct Substitution Solution:")
    print(f"  x1* = {x1_sub:.4f}, x2* = {x2_sub:.4f}, x3* = {x3_sub:.4f}")
    print(f"  f* = {f_sub_val:.4f}")
    
    # 3. SLSQP on full 3-variable problem
    def f_full(x):
        return p * x[0]**2 + q * x[1]**2 + r * x[2]**2
        
    cons = ({'type': 'eq', 'fun': lambda x: x[0] + x[1] + x[2] - C})
    res_slsqp = minimize(f_full, [1.0, 1.0, 1.0], method='SLSQP', constraints=cons)
    x1_sls, x2_sls, x3_sls = res_slsqp.x
    f_sls_val = res_slsqp.fun
    
    print("\n3. SLSQP Numerical Solution:")
    print(f"  x1* = {x1_sls:.4f}, x2* = {x2_sls:.4f}, x3* = {x3_sls:.4f}")
    print(f"  f* = {f_sls_val:.4f}")
    
    # Check agreement to 4 decimals
    match_x1 = np.isclose(x1_lag, x1_sub, atol=1e-4) and np.isclose(x1_lag, x1_sls, atol=1e-4)
    match_x2 = np.isclose(x2_lag, x2_sub, atol=1e-4) and np.isclose(x2_lag, x2_sls, atol=1e-4)
    match_x3 = np.isclose(x3_lag, x3_sub, atol=1e-4) and np.isclose(x3_lag, x3_sls, atol=1e-4)
    match_f  = np.isclose(f_lag, f_sub_val, atol=1e-4) and np.isclose(f_lag, f_sls_val, atol=1e-4)
    
    print(f"\nAll three methods agree to 4 decimals: {match_x1 and match_x2 and match_x3 and match_f}")

if __name__ == "__main__":
    roll_num = sys.argv[1] if len(sys.argv) > 1 else "25027"
    run(roll_num)
