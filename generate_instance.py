import sys, re, json
import numpy as np

def instance(roll):
    digits = re.sub(r"\D", "", str(roll))
    d = digits.zfill(4)[-4:]; d1, d2, d3, d4 = map(int, d)
    p, q, r, k = d4 % 5 + 1, d3 % 4 + 1, d2 % 3 + 1, (d1+d2+d3+d4) % 9 + 1
    rng = np.random.default_rng(int(digits))
    n, m = 40 + 2*k, 12                       # demand points, candidate hubs
    pts = rng.uniform(0, 8, size=(n, 2))      # km
    gx, gy = np.meshgrid([1, 3, 5, 7], [1.3, 4, 6.7])
    sites = np.c_[gx.ravel(), gy.ravel()] + rng.uniform(-0.6, 0.6, size=(m, 2))
    demand = rng.integers(1, 6 + p, size=n)               # deliveries / hour
    cap = rng.integers(25, 46, size=m) + 2*q               # deliveries / hour
    fixed = rng.integers(800, 1501, size=m) + 50*r         # Rs / hour (amortised)
    dist = np.linalg.norm(pts[:, None, :] - sites[None, :, :], axis=2)
    return dict(roll=str(roll), p=p, q=q, r=r, k=k, n=n, m=m,
                c1=0.02, c2=2000*(1 + 0.1*q), E=100 + 10*r, kappa=0.5, gamma=0.1,
                points=pts.round(3).tolist(), sites=sites.round(3).tolist(),
                demand=demand.tolist(), capacity=cap.tolist(), fixed=fixed.tolist(),
                dist=dist.round(3).tolist())

if __name__ == "__main__":
    roll = sys.argv[1] if len(sys.argv) > 1 else "25027"
    inst = instance(roll)
    json.dump(inst, open(f"instance_{inst['roll']}.json", "w"), indent=1)
    print(f"roll {roll}: p={inst['p']} q={inst['q']} r={inst['r']} k={inst['k']}  n={inst['n']} m={inst['m']}")