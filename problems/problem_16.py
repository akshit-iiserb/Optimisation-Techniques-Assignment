import numpy as np
import matplotlib.pyplot as plt

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
    
    c = r + 1.0 + 0.1 * k
    lam_star = c**(1.0 / 3.0)
    print(f"Parameter c = {c:.4f}, Analytical minimizer lambda* = c^(1/3) = {lam_star:.4f}")
    
    def f(lam):
        return lam**4 - 4.0 * c * lam
        
    n = 11
    a0, b0 = 0.0, 3.0
    L0 = b0 - a0
    
    # 1. Fibonacci Search (n = 11)
    F = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]
    
    print("\n--- 1. Fibonacci Method (n = 11) ---")
    a, b = a0, b0
    evals_fib = [2]
    lens_fib = [b - a]
    
    L2_star = (F[n - 2] / F[n]) * L0
    lam1 = a + L2_star
    lam2 = b - L2_star
    f1 = f(lam1)
    f2 = f(lam2)
    lens_fib[0] = b - a
    
    print(f"{'Eval':>4} | {'a':>7} | {'b':>7} | {'L':>7} | {'lam1':>7} | {'lam2':>7} | {'f1':>8} | {'f2':>8}")
    print("-" * 75)
    print(f"{2:4d} | {a:7.4f} | {b:7.4f} | {(b-a):7.4f} | {lam1:7.4f} | {lam2:7.4f} | {f1:8.4f} | {f2:8.4f}")
    
    for k_step in range(2, n):
        if k_step == n - 1:
            eps = 0.001
            lam2 = lam1 + eps
            f2 = f(lam2)
        else:
            if f1 < f2:
                b = lam2
                lam2 = lam1
                f2 = f1
                lam1 = a + (F[n - k_step - 1] / F[n - k_step + 1]) * (b - a)
                f1 = f(lam1)
            else:
                a = lam1
                lam1 = lam2
                f1 = f2
                lam2 = b - (F[n - k_step - 1] / F[n - k_step + 1]) * (b - a)
                f2 = f(lam2)
                
        evals_fib.append(k_step + 1)
        lens_fib.append(b - a)
        print(f"{k_step+1:4d} | {a:7.4f} | {b:7.4f} | {(b-a):7.4f} | {lam1:7.4f} | {lam2:7.4f} | {f1:8.4f} | {f2:8.4f}")
        
    if f1 < f2:
        b = lam2
    else:
        a = lam1
        
    final_fib = [a, b]
    final_L_fib = b - a
    ratio_fib = final_L_fib / L0
    print(f"\nFibonacci Final Interval: [{final_fib[0]:.4f}, {final_fib[1]:.4f}]")
    print(f"Fibonacci Final Length Ln = {final_L_fib:.4f}, Ln/L0 = {ratio_fib:.4f}")

    # 2. Golden Section Search (n = 11)
    print("\n--- 2. Golden Section Method (n = 11) ---")
    a, b = a0, b0
    gamma = (np.sqrt(5.0) - 1.0) / 2.0
    
    lam1 = b - gamma * (b - a)
    lam2 = a + gamma * (b - a)
    f1 = f(lam1)
    f2 = f(lam2)
    
    evals_gs = [2]
    lens_gs = [b - a]
    
    print(f"{'Eval':>4} | {'a':>7} | {'b':>7} | {'L':>7} | {'lam1':>7} | {'lam2':>7} | {'f1':>8} | {'f2':>8}")
    print("-" * 75)
    print(f"{2:4d} | {a:7.4f} | {b:7.4f} | {(b-a):7.4f} | {lam1:7.4f} | {lam2:7.4f} | {f1:8.4f} | {f2:8.4f}")
    
    for k_step in range(3, n + 1):
        if f1 < f2:
            b = lam2
            lam2 = lam1
            f2 = f1
            lam1 = b - gamma * (b - a)
            f1 = f(lam1)
        else:
            a = lam1
            lam1 = lam2
            f1 = f2
            lam2 = a + gamma * (b - a)
            f2 = f(lam2)
            
        evals_gs.append(k_step)
        lens_gs.append(b - a)
        print(f"{k_step:4d} | {a:7.4f} | {b:7.4f} | {(b-a):7.4f} | {lam1:7.4f} | {lam2:7.4f} | {f1:8.4f} | {f2:8.4f}")
        
    final_gs = [a, b]
    final_L_gs = b - a
    ratio_gs = final_L_gs / L0
    print(f"\nGolden Section Final Interval: [{final_gs[0]:.4f}, {final_gs[1]:.4f}]")
    print(f"Golden Section Final Length Ln = {final_L_gs:.4f}, Ln/L0 = {ratio_gs:.4f}")

    # Plotting
    plt.figure(figsize=(8, 5))
    plt.plot(evals_fib, lens_fib, 'bo-', label='Fibonacci Search')
    plt.plot(evals_gs, lens_gs, 'rs--', label='Golden Section Search')
    plt.xlabel('Number of Function Evaluations')
    plt.ylabel('Interval Length')
    plt.title('Problem 16: Interval Reduction Comparison (n = 11)')
    plt.yscale('log')
    plt.grid(True, which='both', linestyle=':')
    plt.legend()
    plt.savefig("figures/problem_16.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved plot to figures/problem_16.png")

if __name__ == "__main__":
    run()
