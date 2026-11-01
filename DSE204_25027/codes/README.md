# DSE 204: Optimization Techniques — Computational Suite

**Student Name:** Akshit Pandey  
**Roll Number:** 25027  
**Parameters:** $p = 3, \quad q = 3, \quad r = 1, \quad k = 6$

---

## 1. Directory Structure

```
codes/
├── main.py                 # Master runner executing all 21 problems sequentially
├── generate_instance.py     # Deterministic instance generator for Problem 21
├── instance_25027.json      # Generated network instance data for Roll 25027
├── output_25027.txt         # Complete console log, tableaus, iteration tables, and outputs
├── requirements.txt         # Standard Python dependency specifications
├── pyproject.toml / uv.lock # Deterministic environment configuration
├── figures/                 # Generated plots and visualization artifacts
│   ├── problem_01.png
│   ├── problem_05.png
│   ├── problem_06.png
│   ├── problem_16.png
│   ├── problem_21_cost_speed.png
│   └── problem_21_lagrangian.png
└── problems/                # Standalone modular implementation for each problem
    ├── problem_01.py        # Single-variable polynomial optimization & grid search
    ├── problem_02.py        # 3-variable unconstrained optimization & Hessian classification
    ├── problem_03.py        # Equality constrained: Lagrange multipliers vs substitution vs SLSQP
    ├── problem_04.py        # Lagrange multiplier sensitivity analysis (tank area)
    ├── problem_05.py        # KKT conditions, active sets & contour visualization
    ├── problem_06.py        # Graphical LP method & vertex enumeration
    ├── problem_07.py        # Standard form conversion & basic solutions enumeration
    ├── problem_08.py        # Custom simplex method implementation from scratch
    ├── problem_09.py        # Two-phase simplex method (Phase I artificial & Phase II)
    ├── problem_10.py        # Simplex special cases (unbounded, alternative optima, infeasible)
    ├── problem_11.py        # Dual LP formulation & complementary slackness verification
    ├── problem_12.py        # Shadow prices & economic interpretation from tableau
    ├── problem_13.py        # Dual simplex method implementation
    ├── problem_14.py        # Accelerated step search & exhaustive line search
    ├── problem_15.py        # Dichotomous search & interval halving line search
    ├── problem_16.py        # Fibonacci search vs Golden Section search comparison
    ├── problem_17.py        # Quadratic interpolation (Rao's procedure with refitting)
    ├── problem_18.py        # Cubic Hermite interpolation with derivative evaluations
    ├── problem_19.py        # Pure integer programming via depth-first Branch & Bound
    ├── problem_20.py        # 0–1 Knapsack problem via greedy Branch & Bound
    └── problem_21.py        # Drone delivery network: speed, LP, B&B, Lagrangian & cost optimization
```

---

## 2. Dependencies & Installation

The code requires Python 3.10+ with standard scientific libraries:
* `numpy >= 1.24.0`
* `scipy >= 1.10.0`
* `matplotlib >= 3.7.0`

### Option A: Using standard pip
```bash
pip install -r requirements.txt
```

### Option B: Using uv (Recommended for deterministic reproduction)
```bash
uv sync
```

---

## 3. Running the Code

### Execute the complete suite:
To run the full suite for Roll Number 25027 and regenerate all outputs:
```bash
python main.py 25027
```
*(Or with `uv`: `uv run python main.py 25027`)*

### Run any individual problem:
Each problem script is self-contained and runnable independently:
```bash
python problems/problem_08.py
python problems/problem_21.py
```

---

## 4. Output Logs & Figures

* **`output_25027.txt`**: Contains the complete run trace, including all simplex tableaus, iteration tables, convergence metrics, and numerical verifications.
* **`figures/`**: Contains all generated plots, including:
  * `problem_01.png`: Quartic function and stationary points.
  * `problem_05.png`: Feasible region contours and optimal point.
  * `problem_06.png`: Graphical LP polytope, constraints, and vertices.
  * `problem_16.png`: Interval reduction rate ($L_n/L_0$) for Fibonacci vs Golden Section.
  * `problem_21_lagrangian.png`: Lagrangian lower and upper bound convergence progression.
  * `problem_21_cost_speed.png`: Delivery cost curve $F(v)$ vs cruise speed.

