import sys
import re
import numpy as np
from scipy.optimize import minimize_scalar

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
    
    eval_count = 0
    def f(lam):
        nonlocal eval_count
        eval_count += 1
        return lam**4 - (3.0 + p) * lam**2 - (2.0 + q) * lam + 5.0
        
    # Phase 1: Rao's bracketing procedure starting at A = 0, t0 = 0.5
    A = 0.0
    fA = f(A)
    t = 0.5
    pts = [A]
    f_vals = [fA]
    
    curr = A + t
    f_curr = f(curr)
    pts.append(curr)
    f_vals.append(f_curr)
    
    while f_curr < f_vals[-2]:
        t *= 2.0
        curr = pts[-1] + t
        f_curr = f(curr)
        pts.append(curr)
        f_vals.append(f_curr)
        
    # The last 3 points bracket the minimum: A, B, C
    x1, x2, x3 = pts[-3], pts[-2], pts[-1]
    f1, f2, f3 = f_vals[-3], f_vals[-2], f_vals[-1]
    
    print("\nInitial Bracketing Triplet:")
    print(f"  A = {x1:.4f}, f(A) = {f1:.4f}")
    print(f"  B = {x2:.4f}, f(B) = {f2:.4f}")
    print(f"  C = {x3:.4f}, f(C) = {f3:.4f}")
    
    print("\n--- Quadratic Interpolation Iterations ---")
    print(f"{'Fit':>4} | {'A':>7} | {'B':>7} | {'C':>7} | {'lam_star':>9} | {'h(lam*)':>9} | {'f(lam*)':>9} | {'Test Ratio':>10}")
    print("-" * 80)
    
    fit_idx = 0
    while True:
        fit_idx += 1
        # Parabolic fit formulas
        denom = (x2 - x3) * f1 + (x3 - x1) * f2 + (x1 - x2) * f3
        numer = (x2**2 - x3**2) * f1 + (x3**2 - x1**2) * f2 + (x1**2 - x2**2) * f3
        lam_star = 0.5 * numer / denom
        
        # Polynomial coefficients h(x) = a + b*x + c*x^2
        c_poly = ((f3 - f2) / (x3 - x2) - (f2 - f1) / (x2 - x1)) / (x3 - x1)
        b_poly = (f2 - f1) / (x2 - x1) - c_poly * (x1 + x2)
        a_poly = f1 - b_poly * x1 - c_poly * (x1**2)
        
        h_val = a_poly + b_poly * lam_star + c_poly * (lam_star**2)
        f_val = f(lam_star)
        
        test_val = abs((h_val - f_val) / f_val)
        print(f"{fit_idx:4d} | {x1:7.4f} | {x2:7.4f} | {x3:7.4f} | {lam_star:9.4f} | {h_val:9.4f} | {f_val:9.4f} | {test_val:10.6f}")
        
        if test_val <= 0.001 or fit_idx >= 15:
            break
            
        # Refit: take best point and two neighbours
        all_pts = [(x1, f1), (x2, f2), (x3, f3), (lam_star, f_val)]
        all_pts.sort(key=lambda item: item[0])
        best_i = int(np.argmin([p[1] for p in all_pts]))
        
        # Ensure interior best point
        if best_i == 0:
            triplet = all_pts[:3]
        elif best_i == 3:
            triplet = all_pts[1:]
        else:
            triplet = all_pts[best_i - 1 : best_i + 2]
            
        x1, f1 = triplet[0]
        x2, f2 = triplet[1]
        x3, f3 = triplet[2]
        
    print(f"\nFinal Quadratic Interpolation Result: lambda* = {lam_star:.4f}, f(lambda*) = {f_val:.4f}")
    print(f"Total function evaluations used: {eval_count}")
    
    # Comparison with SciPy minimize_scalar
    res_scipy = minimize_scalar(lambda x: x**4 - (3.0 + p) * x**2 - (2.0 + q) * x + 5.0)
    print(f"\nSciPy minimize_scalar Result:")
    print(f"  lambda* = {res_scipy.x:.4f}, f(lambda*) = {res_scipy.fun:.4f}")
    print(f"  Absolute difference in lambda*: {abs(lam_star - res_scipy.x):.6f}")
    
    print("\nComment on Stopping Test:")
    print("The function-value test |(h - f)/f| <= 0.001 measures relative agreement in function value,")
    print("which can be satisfied even when the predicted point is still somewhat distant from the true root if the function is flat.")

if __name__ == "__main__":
    roll_num = sys.argv[1] if len(sys.argv) > 1 else "25027"
    run(roll_num)
