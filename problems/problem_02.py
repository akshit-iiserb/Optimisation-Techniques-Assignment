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
    
    def f(x):
        return x[0]**3 + x[1]**3 - 3 * (p**2) * x[0] - 3 * (q**2) * x[1] + (x[2] - k)**2 + r
        
    def hessian(x):
        return np.diag([6.0 * x[0], 6.0 * x[1], 2.0])
        
    stat_points = [
        np.array([float(p), float(q), float(k)]),
        np.array([float(p), float(-q), float(k)]),
        np.array([float(-p), float(q), float(k)]),
        np.array([float(-p), float(-q), float(k)])
    ]
    
    print("\nStationary points and Hessian classification:")
    for idx, pt in enumerate(stat_points, 1):
        H = hessian(pt)
        eigvals = np.linalg.eigvalsh(H)
        if np.all(eigvals > 0):
            pt_type = "local minimum"
        elif np.all(eigvals < 0):
            pt_type = "local maximum"
        else:
            pt_type = "saddle point"
        print(f"Point {idx}: ({pt[0]:.4f}, {pt[1]:.4f}, {pt[2]:.4f})")
        print(f"  Hessian eigenvalues: [{eigvals[0]:.4f}, {eigvals[1]:.4f}, {eigvals[2]:.4f}] -> {pt_type}")
        print(f"  f(x) = {f(pt):.4f}")
        
    # 5 random starts for scipy.optimize.minimize
    rng = np.random.default_rng(int(re.sub(r"\D", "", str(roll))))
    print("\nRunning scipy.optimize.minimize from 5 random starts:")
    for i in range(5):
        # Sample starting points around the stationary points
        x0 = rng.uniform(-2 * max(p, q), 2 * max(p, q), size=3)
        x0[2] = rng.uniform(k - 5, k + 5)
        res = minimize(f, x0, method='BFGS')
        
        reached_label = "diverged / other"
        if res.success:
            for idx, pt in enumerate(stat_points, 1):
                if np.allclose(res.x, pt, atol=1e-2):
                    reached_label = f"Point {idx} ({pt[0]:.0f}, {pt[1]:.0f}, {pt[2]:.0f})"
                    break
        print(f"Run {i+1}:")
        print(f"  Start:   [{x0[0]:.4f}, {x0[1]:.4f}, {x0[2]:.4f}]")
        print(f"  Reached: [{res.x[0]:.4f}, {res.x[1]:.4f}, {res.x[2]:.4f}]")
        print(f"  Classification: {reached_label} (status={res.status})")

if __name__ == "__main__":
    roll_num = sys.argv[1] if len(sys.argv) > 1 else "25027"
    run(roll_num)
