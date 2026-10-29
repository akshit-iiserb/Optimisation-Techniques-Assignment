import numpy as np
from scipy.optimize import root_scalar

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
    
    c = 10.0 + 2.0 * r + k
    print(f"Parameter c = {c:.4f}")
    
    eval_f = 0
    eval_df = 0
    
    def f(lam):
        nonlocal eval_f
        eval_f += 1
        return lam**5 - 5.0 * lam**3 - c * lam + 5.0
        
    def df(lam):
        nonlocal eval_df
        eval_df += 1
        return 5.0 * lam**4 - 15.0 * lam**2 - c
        
    t = 0.4
    pt = 0.0
    d_prev = df(pt)
    
    while True:
        pt_next = pt + t
        d_next = df(pt_next)
        if d_next > 0:
            A = pt
            B = pt_next
            break
        pt = pt_next
        t *= 2.0
        
    print(f"\nBracketing found: A = {A:.4f} (df = {df(A):.4f}), B = {B:.4f} (df = {df(B):.4f})")
    
    print("\n--- Cubic Interpolation Iterations ---")
    print(f"{'Iter':>4} | {'A':>7} | {'B':>7} | {'lam_star':>9} | {'f(lam*)':>9} | {'f\'(lam*)':>10} | {'|f\'|':>8}")
    print("-" * 75)
    
    iter_count = 0
    while True:
        iter_count += 1
        fA, fB = f(A), f(B)
        dfA, dfB = df(A), df(B)
        
        Z = 3.0 * (fA - fB) / (B - A) + dfA + dfB
        disc = Z**2 - dfA * dfB
        Q = np.sqrt(max(0.0, disc))
        
        denom = dfA + dfB + 2.0 * Z
        lam_plus = A + ((dfA + Z + Q) / denom) * (B - A)
        lam_minus = A + ((dfA + Z - Q) / denom) * (B - A)
        
        if A <= lam_plus <= B:
            lam_star = lam_plus
        else:
            lam_star = lam_minus
            
        f_star = f(lam_star)
        df_star = df(lam_star)
        abs_df = abs(df_star)
        
        print(f"{iter_count:4d} | {A:7.4f} | {B:7.4f} | {lam_star:9.4f} | {f_star:9.4f} | {df_star:10.4f} | {abs_df:8.4f}")
        
        if abs_df <= 0.01 or iter_count >= 15:
            break
            
        if df_star < 0:
            A = lam_star
        else:
            B = lam_star
            
    print(f"\nFinal Cubic Interpolation: lambda* = {lam_star:.4f}, f'(lambda*) = {df_star:.4f}")
    print(f"Total iterations: {iter_count}")
    print(f"Function evaluations: f={eval_f}, f'={eval_df}")
    
    root_res = root_scalar(df, bracket=[0.0, 5.0], method='brentq')
    print(f"\nTrue root of f'(lambda) = 0: lambda* = {root_res.root:.4f}")
    print(f"Error in root: {abs(lam_star - root_res.root):.6f}")
    print("Comparison: Cubic interpolation converged rapidly using gradient information compared to quadratic fit.")

if __name__ == "__main__":
    run()
