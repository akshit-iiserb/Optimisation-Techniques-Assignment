import sys
import re
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog

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
    
    # Lines: a1*x1 + a2*x2 = b
    # 0: x1 + 2*x2 = 10 + r
    # 1: 3*x1 + x2 = 15 + p
    # 2: x1 = 4 + q
    # 3: x1 = 0
    # 4: x2 = 0
    lines = [
        (np.array([1.0, 2.0]), 10.0 + r),
        (np.array([3.0, 1.0]), 15.0 + p),
        (np.array([1.0, 0.0]), 4.0 + q),
        (np.array([1.0, 0.0]), 0.0),
        (np.array([0.0, 1.0]), 0.0)
    ]
    
    def is_feasible(pt):
        x1, x2 = pt
        if x1 < -1e-7 or x2 < -1e-7:
            return False
        if x1 + 2.0 * x2 > 10.0 + r + 1e-7:
            return False
        if 3.0 * x1 + x2 > 15.0 + p + 1e-7:
            return False
        if x1 > 4.0 + q + 1e-7:
            return False
        return True

    feasible_vertices = []
    
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            A = np.array([lines[i][0], lines[j][0]])
            b = np.array([lines[i][1], lines[j][1]])
            if abs(np.linalg.det(A)) > 1e-9:
                pt = np.linalg.solve(A, b)
                if is_feasible(pt):
                    # Check if already present
                    already = False
                    for v in feasible_vertices:
                        if np.allclose(v, pt, atol=1e-5):
                            already = True
                            break
                    if not already:
                        feasible_vertices.append(pt)
                        
    print("\nFeasible Vertices and Objective Values:")
    best_z = -1e9
    best_vertex = None
    for v in feasible_vertices:
        z = (p + 2.0) * v[0] + (q + 1.0) * v[1]
        print(f"  Vertex ({v[0]:.4f}, {v[1]:.4f}) -> z = {z:.4f}")
        if z > best_z:
            best_z = z
            best_vertex = v
            
    print(f"\nOptimal Vertex by Graphical Method: ({best_vertex[0]:.4f}, {best_vertex[1]:.4f}) with z = {best_z:.4f}")
    
    # Confirm with scipy.optimize.linprog
    c_obj = [-(p + 2.0), -(q + 1.0)]
    A_ub = [
        [1.0, 2.0],
        [3.0, 1.0],
        [1.0, 0.0]
    ]
    b_ub = [10.0 + r, 15.0 + p, 4.0 + q]
    bounds = [(0.0, None), (0.0, None)]
    
    res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    print(f"Linprog Confirmation:")
    print(f"  x1* = {res.x[0]:.4f}, x2* = {res.x[1]:.4f}, z* = {-res.fun:.4f}")
    
    # Plotting
    os.makedirs("figures", exist_ok=True)
    plt.figure(figsize=(8, 6))
    
    x1_vals = np.linspace(0, 4.0 + q + 2.0, 400)
    plt.plot(x1_vals, (10.0 + r - x1_vals) / 2.0, 'r-', label=r'$x_1 + 2x_2 \leq 10+r$')
    plt.plot(x1_vals, 15.0 + p - 3.0 * x1_vals, 'g-', label=r'$3x_1 + x_2 \leq 15+p$')
    plt.axvline(x=4.0 + q, color='b', linestyle='-', label=r'$x_1 \leq 4+q$')
    
    # Sort vertices counter-clockwise
    verts = np.array(feasible_vertices)
    center = np.mean(verts, axis=0)
    angles = np.arctan2(verts[:, 1] - center[1], verts[:, 0] - center[0])
    sorted_idx = np.argsort(angles)
    sorted_verts = verts[sorted_idx]
    
    plt.fill(sorted_verts[:, 0], sorted_verts[:, 1], 'lightgray', alpha=0.5, label='Feasible Region')
    for v in feasible_vertices:
        plt.plot(v[0], v[1], 'ko')
        plt.text(v[0] + 0.1, v[1] + 0.1, f"({v[0]:.2f}, {v[1]:.2f})")
        
    plt.plot(best_vertex[0], best_vertex[1], 'r*', markersize=12, label='Optimum')
    plt.xlim(-0.5, 4.0 + q + 1.0)
    plt.ylim(-0.5, max(10.0 + r, 15.0 + p) / 2.0 + 2.0)
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")
    plt.title("Problem 6: Graphical LP Method")
    plt.legend()
    plt.grid(True)
    plt.savefig("figures/problem_06.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved plot to figures/problem_06.png")

if __name__ == "__main__":
    roll_num = sys.argv[1] if len(sys.argv) > 1 else "25027"
    run(roll_num)
