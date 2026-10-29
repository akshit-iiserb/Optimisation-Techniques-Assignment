import sys
import re
import os
import numpy as np
import matplotlib.pyplot as plt
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
    
    def obj(x):
        return (x[0] - p - 1.0)**2 + (x[1] - q - 1.0)**2
        
    def grad_obj(x):
        return np.array([2.0 * (x[0] - p - 1.0), 2.0 * (x[1] - q - 1.0)])
        
    cons = ({'type': 'ineq', 'fun': lambda x: r - (x[0] + x[1])})
    bounds = [(0.0, None), (0.0, None)]
    
    res = minimize(obj, [0.0, 0.0], method='SLSQP', bounds=bounds, constraints=cons, tol=1e-9)
    x1_star, x2_star = res.x
    f_star = res.fun
    
    print(f"\nSLSQP Solution:")
    print(f"  x1* = {x1_star:.4f}")
    print(f"  x2* = {x2_star:.4f}")
    print(f"  f*  = {f_star:.4f}")
    
    # Stationarity: grad f(x*) + mu1*[1, 1]^T - mu2*[1, 0]^T - mu3*[0, 1]^T = 0
    # => [1, -1, 0; 1, 0, -1] * [mu1, mu2, mu3]^T = -grad f(x*)
    grad = grad_obj(res.x)
    
    # Active constraints check
    mu1, mu2, mu3 = 0.0, 0.0, 0.0
    if np.isclose(x1_star + x2_star, r, atol=1e-4):
        if x1_star > 1e-4 and x2_star > 1e-4:
            mu1 = -grad[0]
            mu2 = 0.0
            mu3 = 0.0
        elif np.isclose(x1_star, 0.0, atol=1e-4):
            mu1 = -grad[1]
            mu2 = -grad[0] - mu1
            mu3 = 0.0
        else:
            mu1 = -grad[0]
            mu2 = 0.0
            mu3 = -grad[1] - mu1
            
    print("\nRecovered KKT Multipliers:")
    print(f"  mu1 (x1 + x2 <= r): {mu1:.4f}")
    print(f"  mu2 (x1 >= 0):      {mu2:.4f}")
    print(f"  mu3 (x2 >= 0):      {mu3:.4f}")
    
    # Plotting
    os.makedirs("figures", exist_ok=True)
    plt.figure(figsize=(7, 6))
    
    # Grid for contours
    X1, X2 = np.meshgrid(np.linspace(0, max(p + 2, r + 1), 200), np.linspace(0, max(q + 2, r + 1), 200))
    Z = (X1 - p - 1.0)**2 + (X2 - q - 1.0)**2
    cs = plt.contour(X1, X2, Z, levels=25, cmap='viridis')
    plt.clabel(cs, inline=1, fontsize=8)
    
    # Feasible region
    poly_x = [0.0, r, 0.0, 0.0]
    poly_y = [0.0, 0.0, r, 0.0]
    plt.fill(poly_x, poly_y, color='skyblue', alpha=0.5, label='Feasible region')
    plt.plot([0, r], [r, 0], 'b-', linewidth=2, label=r'$x_1 + x_2 \leq r$')
    
    plt.plot(x1_star, x2_star, 'ro', markersize=8, label=f'Optimum ({x1_star:.2f}, {x2_star:.2f})')
    plt.plot(p + 1.0, q + 1.0, 'k*', markersize=10, label=f'Target ({p+1}, {q+1})')
    
    plt.title("Problem 5: KKT Conditions & Feasible Contours")
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")
    plt.xlim(-0.2, max(p + 2, r + 1))
    plt.ylim(-0.2, max(q + 2, r + 1))
    plt.grid(True)
    plt.legend()
    plt.savefig("figures/problem_05.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved plot to figures/problem_05.png")

if __name__ == "__main__":
    roll_num = sys.argv[1] if len(sys.argv) > 1 else "25027"
    run(roll_num)
