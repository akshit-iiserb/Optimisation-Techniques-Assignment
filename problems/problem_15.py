import sys
import re
import numpy as np

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
    
    c = 3.0 + q + 0.1 * k
    lam_star = np.log(c)
    print(f"Parameter c = {c:.4f}, Analytical minimizer lambda* = ln(c) = {lam_star:.4f}")
    
    def f(lam):
        return np.exp(lam) - c * lam

    # 1. Dichotomous Search
    print("\n--- 1. Dichotomous Search (delta = 0.001, tol = 0.01) ---")
    a, b = 0.0, 3.0
    delta = 0.001
    evals_dich = 0
    iter_dich = 0
    
    print(f"{'Iter':>4} | {'a':>7} | {'b':>7} | {'L':>7} | {'lam1':>7} | {'lam2':>7} | {'f(lam1)':>8} | {'f(lam2)':>8} | Discarded")
    print("-" * 80)
    
    while (b - a) > 0.01:
        iter_dich += 1
        mid = (a + b) / 2.0
        lam1 = mid - delta / 2.0
        lam2 = mid + delta / 2.0
        f1 = f(lam1)
        f2 = f(lam2)
        evals_dich += 2
        
        if f1 < f2:
            discarded = f"[{lam2:.4f}, {b:.4f}]"
            b = lam2
        else:
            discarded = f"[{a:.4f}, {lam1:.4f}]"
            a = lam1
            
        print(f"{iter_dich:4d} | {a:7.4f} | {b:7.4f} | {(b-a):7.4f} | {lam1:7.4f} | {lam2:7.4f} | {f1:8.4f} | {f2:8.4f} | {discarded}")
        
    final_mid_dich = (a + b) / 2.0
    error_dich = abs(final_mid_dich - lam_star)
    print(f"\nDichotomous Search Results:")
    print(f"  Final interval: [{a:.4f}, {b:.4f}], Length = {b - a:.4f}")
    print(f"  Estimated lambda* = {final_mid_dich:.4f}")
    print(f"  Evaluations used  = {evals_dich}")
    print(f"  Error against ln(c) = {error_dich:.6f}")

    # 2. Interval Halving
    print("\n--- 2. Interval Halving (tol = 0.01) ---")
    a, b = 0.0, 3.0
    xm = (a + b) / 2.0
    fm = f(xm)
    evals_halv = 1
    iter_halv = 0
    
    print(f"{'Iter':>4} | {'a':>7} | {'b':>7} | {'L':>7} | {'x1':>7} | {'xm':>7} | {'x2':>7} | {'f1':>8} | {'fm':>8} | {'f2':>8} | Discarded")
    print("-" * 95)
    
    while (b - a) > 0.01:
        iter_halv += 1
        L = b - a
        x1 = a + L / 4.0
        x2 = b - L / 4.0
        f1 = f(x1)
        f2 = f(x2)
        evals_halv += 2
        
        if f1 < fm:
            discarded = f"[{xm:.4f}, {b:.4f}]"
            b = xm
            xm = x1
            fm = f1
        elif f2 < fm:
            discarded = f"[{a:.4f}, {xm:.4f}]"
            a = xm
            xm = x2
            fm = f2
        else:
            discarded = f"[{a:.4f}, {x1:.4f}] & [{x2:.4f}, {b:.4f}]"
            a = x1
            b = x2
            
        print(f"{iter_halv:4d} | {a:7.4f} | {b:7.4f} | {(b-a):7.4f} | {x1:7.4f} | {xm:7.4f} | {x2:7.4f} | {f1:8.4f} | {fm:8.4f} | {f2:8.4f} | {discarded}")
        
    final_mid_halv = xm
    error_halv = abs(final_mid_halv - lam_star)
    print(f"\nInterval Halving Results:")
    print(f"  Final interval: [{a:.4f}, {b:.4f}], Length = {b - a:.4f}")
    print(f"  Estimated lambda* = {final_mid_halv:.4f}")
    print(f"  Evaluations used  = {evals_halv}")
    print(f"  Error against ln(c) = {error_halv:.6f}")

if __name__ == "__main__":
    roll_num = sys.argv[1] if len(sys.argv) > 1 else "25027"
    run(roll_num)
