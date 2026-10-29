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
    
    a = p
    b = p + q
    
    def f(x):
        return 3 * x**4 - 4 * (a + b) * x**3 + 6 * a * b * x**2 + r
        
    def f2(x):
        return 12 * (3 * x**2 - 2 * (a + b) * x + a * b)
        
    stat_pts = [0.0, float(a), float(b)]
    print("Stationary points:")
    for pt in stat_pts:
        val = f(pt)
        d2 = f2(pt)
        if d2 > 0:
            classification = "local minimum"
        elif d2 < 0:
            classification = "local maximum"
        else:
            classification = "inconclusive"
        print(f"  x = {pt:.4f}, f(x) = {val:.4f}, f''(x) = {d2:.4f} -> {classification}")
        
    if np.isclose(f(0.0), f(float(b))):
        print(f"Analytical global minimum over R: occurs at both x = 0.0000 and x = {float(b):.4f}, f(x) = {f(0.0):.4f}")
    elif f(0.0) < f(float(b)):
        print(f"Analytical global minimum over R: x = 0.0000, f(x) = {f(0.0):.4f}")
    else:
        print(f"Analytical global minimum over R: x = {float(b):.4f}, f(x) = {f(float(b)):.4f}")
    
    # Fine grid search
    grid_x = np.linspace(-1.0, b + 1.0, 1000000)
    grid_y = f(grid_x)
    min_idx = np.argmin(grid_y)
    grid_min_x = grid_x[min_idx]
    grid_min_val = grid_y[min_idx]
    print(f"Grid search minimum: x = {grid_min_x:.4f}, f(x) = {grid_min_val:.4f}")
    
    # Plotting
    x_plot = np.linspace(-1.0, b + 1.0, 1000)
    plt.figure(figsize=(8, 5))
    plt.plot(x_plot, f(x_plot), 'b-', label='f(x)')
    for pt in stat_pts:
        plt.plot(pt, f(pt), 'ro')
        plt.text(pt, f(pt), f" ({pt:.1f}, {f(pt):.1f})")
    plt.title("Problem 1: Single-variable optimum")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.grid(True)
    plt.legend()
    plt.savefig("figures/problem_01.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved plot to figures/problem_01.png")

if __name__ == "__main__":
    run()
