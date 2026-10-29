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
    
    # Analytical minimizer
    lam_star = (4.0 + 2.0 * p)**(1.0 / 3.0)
    print(f"\nAnalytical minimizer lambda* = (4 + 2p)^(1/3) = {lam_star:.4f}")
    
    eval_count = 0
    def f(lam):
        nonlocal eval_count
        eval_count += 1
        return 2.0 * lam**2 + (16.0 + 8.0 * p) / lam
        
    # 1. Unrestricted search with accelerated step
    print("\n--- 1. Unrestricted Search (Accelerated Step) ---")
    eval_count = 0
    lam_pts = [0.5]
    f_vals = [f(0.5)]
    s = 0.1
    
    print(f"{'Iter':>5} | {'lambda':>8} | {'step s':>8} | {'f(lambda)':>10}")
    print("-" * 40)
    print(f"{0:>5} | {lam_pts[0]:8.4f} | {'-':>8} | {f_vals[0]:10.4f}")
    
    step_idx = 1
    while True:
        lam_next = lam_pts[-1] + s
        f_next = f(lam_next)
        lam_pts.append(lam_next)
        f_vals.append(f_next)
        print(f"{step_idx:>5} | {lam_next:8.4f} | {s:8.4f} | {f_next:10.4f}")
        
        if f_next > f_vals[-2]:
            bracket = [lam_pts[-3] if len(lam_pts) >= 3 else lam_pts[0], lam_pts[-1]]
            break
        s *= 2.0
        step_idx += 1
        
    print(f"\nBracket found: [{bracket[0]:.4f}, {bracket[1]:.4f}], Length = {bracket[1] - bracket[0]:.4f}")
    print(f"Function evaluations used: {eval_count}")
    
    # 2. Exhaustive search on [0.5, 5.0] with n=8
    print("\n--- 2. Exhaustive Search (n = 8) on [0.5, 5.0] ---")
    eval_count = 0
    a, b = 0.5, 5.0
    L0 = b - a
    n1 = 8
    delta1 = L0 / (n1 + 1.0)
    grid_lam1 = [a + i * delta1 for i in range(n1 + 2)]
    grid_f1 = [f(lam) for lam in grid_lam1]
    
    print(f"{'Point':>5} | {'lambda':>8} | {'f(lambda)':>10}")
    print("-" * 30)
    for i in range(len(grid_lam1)):
        print(f"{i:>5} | {grid_lam1[i]:8.4f} | {grid_f1[i]:10.4f}")
        
    min_i1 = int(np.argmin(grid_f1))
    final_int1 = [grid_lam1[max(0, min_i1 - 1)], grid_lam1[min(len(grid_lam1) - 1, min_i1 + 1)]]
    print(f"Minimum at point {min_i1} (lambda = {grid_lam1[min_i1]:.4f})")
    print(f"Final interval: [{final_int1[0]:.4f}, {final_int1[1]:.4f}], Length = {final_int1[1] - final_int1[0]:.4f}")
    print(f"Function evaluations used: {eval_count}")
    
    # 3. Smallest n for final interval <= 1% of L0
    # Final length is 2*L0 / (n + 1) <= 0.01 * L0 => 2/(n+1) <= 0.01 => n + 1 >= 200 => n >= 199
    n_star = int(np.ceil(2.0 / 0.01 - 1.0))
    print(f"\n--- 3. Exhaustive Search for <= 1% interval ---")
    print(f"Smallest n = {n_star}")
    
    eval_count = 0
    delta_star = L0 / (n_star + 1.0)
    grid_lam_star = [a + i * delta_star for i in range(n_star + 2)]
    grid_f_star = [f(lam) for lam in grid_lam_star]
    
    min_i_star = int(np.argmin(grid_f_star))
    final_int_star = [grid_lam_star[min_i_star - 1], grid_lam_star[min_i_star + 1]]
    final_len_star = final_int_star[1] - final_int_star[0]
    
    print(f"Final interval: [{final_int_star[0]:.4f}, {final_int_star[1]:.4f}]")
    print(f"Final length = {final_len_star:.4f} ({final_len_star / L0 * 100:.2f}% of L0)")
    print(f"Function evaluations used: {eval_count}")
    print(f"True minimizer lambda* = {lam_star:.4f} is inside interval: {final_int_star[0] <= lam_star <= final_int_star[1]}")

if __name__ == "__main__":
    roll_num = sys.argv[1] if len(sys.argv) > 1 else "25027"
    run(roll_num)
