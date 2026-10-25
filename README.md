# Applied Optimization — Assignment Suite

## Assignment Overview

This repository contains the complete programming and report structure for the **Optimization Techniques / Applied Optimization** course assignment (150 Marks).

The assignment is uniquely personalized for every student based on their institutional roll number. All problem parameters, constraints, and datasets are derived deterministically from the student's roll number digits.

### Dual Submission Requirements
- **Handwritten (H) [2 Marks per problem]:** Analytical setups, first 1–2 iterations, initial and intermediate tableaus with student-specific numbers substituted.
- **Code (C) [3 Marks per problem]:** Python implementation reading the student roll number, calculating $(p, q, r, k)$, printing the parameter banner, and executing numerical routines.
- **Consistency Rule:** Handwritten numerical values must strictly agree with the first iterations produced by the Python scripts.
- **Board Viva Rule:** Randomly selected students will be asked to reproduce handwritten steps with fresh numbers at the board.
- **Numerical Reporting:** All floating-point answers must be reported to **4 decimal places**.
- **Deliverables:**
  1. A formal report written in **LaTeX**.
  2. A self-contained code repository runnable via `python main.py <roll>`, which automatically regenerates every table and figure.

---

## Assignment Topics & Problem Breakdown

The 21 problems span 5 major areas of mathematical optimization:

### Part 1: Classical & Nonlinear Optimization (Problems 1–5)
- **Problem 1 — Single-Variable Optimum:** Stationary points of a quartic polynomial, second-derivative classification ($f''(x)$), global minimum over $\mathbb{R}$, function plot on $[-1, b+1]$, and confirmation via fine grid search.
- **Problem 2 — Multivariable Unconstrained Optimization:** Stationary points of a 3-variable cubic function, Hessian construction, principal-minor definiteness test, numerical eigenvalues, and multi-start local searches (`scipy.optimize.minimize`).
- **Problem 3 — Equality Constrained Optimization:** Quadratic objective with linear equality constraint. Analytical solution via Lagrange multipliers with bordered-Hessian sufficiency test, comparison against direct substitution (variable elimination) and SLSQP numerical solver.
- **Problem 4 — Lagrange Multipliers as Sensitivity:** Sheet area minimization of a closed cylindrical tank subject to volume constraint $V_0 = 50(p+q)\,\text{m}^3$. Prediction of area change $\Delta S^*$ for unit volume increase using optimal multiplier $\lambda^*$ vs. exact numerical re-solving at $V_0 + 1$.
- **Problem 5 — Karush-Kuhn-Tucker (KKT) Conditions:** Convex quadratic objective under linear inequality and non-negativity constraints ($x_1 + x_2 \le r, x_1 \ge 0, x_2 \ge 0$). Active set identification, analytical multiplier derivation, SLSQP verification with multiplier recovery, and 2D contour plot with feasible set overlay.

### Part 2: Linear Programming & Simplex Methods (Problems 6–13)
- **Problem 6 — Graphical LP Method:** 2-variable LP maximization, geometric plotting of feasible polytope, vertex identification, binding constraint determination, and verification with `scipy.optimize.linprog`.
- **Problem 7 — Standard Form & Basic Solutions:** Conversion of general LP to standard equality form (slack/surplus variables, sign-restricted variables, RHS positivity). Enumeration of all 6 basic solutions, feasibility classification, and BFS optimality identification.
- **Problem 8 — Custom Simplex Method Implementation:** From-scratch tableau simplex implementation. Entering variable selection, minimum ratio test for leaving variable, pivot updates, and iterative tableau printing.
- **Problem 9 — Two-Phase Simplex Method:** Implementation of two-phase simplex for general constraints ($\ge, \le, =$) using artificial variables. Phase I artificial objective minimization, basis cleanup, Phase II optimization, and tableau logging.
- **Problem 10 — Special Cases in Simplex:** Algorithmic diagnosis of degeneracy, unboundedness (column with no positive pivot entry), alternative optimal solutions (zero reduced cost for non-basic variable, parametric ray/segment), and infeasibility (artificial variable remaining positive in basis).
- **Problem 11 — Duality of General Linear Programs:** Primal-to-dual conversion rules, dual formulation, simultaneous primal-dual solution using `linprog`, strong duality check ($z^* = w^*$), and complementary slackness verification.
- **Problem 12 — Shadow Prices & Economic Interpretation:** Reading dual prices directly from slack columns of Problem 8 final tableau. Verifying $\Delta z^* = y_1^*$ when RHS increases by 1, and mathematical explanation for why non-binding resources have zero shadow price.
- **Problem 13 — Dual Simplex Method:** Dual simplex implementation starting from dual-feasible but primal-infeasible tableaus. Negative RHS pivot selection, dual ratio test, and iteration to optimality.

### Part 3: One-Dimensional Line Search & Interpolation (Problems 14–18)
*All methods log an iteration table (interval/points, function evaluations, interval width) and total function evaluation count.*
- **Problem 14 — Unrestricted & Exhaustive Search:** Step-doubling accelerated search ($s = 0.1$) for minimum bracketing, followed by equally-spaced exhaustive search achieving $\le 1\%$ interval reduction.
- **Problem 15 — Dichotomous Search & Interval Halving:** Interval reduction for exponential function $f(\lambda) = e^\lambda - c\lambda$ down to tolerance $\le 0.01$. Evaluation efficiency and comparison with analytical root $\lambda^* = \ln c$.
- **Problem 16 — Fibonacci Search vs. Golden Section Search:** Comparative convergence on quartic function with $n = 11$ function evaluations. Interval length reduction plot ($L_n/L_0$) vs. evaluation count.
- **Problem 17 — Quadratic Interpolation (Rao's Procedure):** 3-point bracketing, parabolic fit $h(\lambda) = a + b\lambda + c\lambda^2$, predicted minimizer $\tilde{\lambda}^* = -b/(2c)$, relative function-value stopping test, and iterative refitting.
- **Problem 18 — Cubic Interpolation (Rao's Method):** Derivative-based bracketing, cubic Hermite interpolation using function values and gradients, iterative reduction until $|f'(\tilde{\lambda}^*)| \le 0.01$.

### Part 4: Integer Programming & Branch-and-Bound (Problems 19–20)
- **Problem 19 — Pure Integer Program:** 2-variable integer LP maximization. Graphical LP relaxation, graphical branch-and-bound tree to depth 2, custom depth-first branch-and-bound implementation using `linprog`, and verification via integer lattice enumeration.
- **Problem 20 — 0–1 Knapsack Problem:** 6-item knapsack problem. Greedy fractional LP relaxation, branch-and-bound search tree, custom B&B algorithm with bounding, node count analysis, and comparison with brute-force enumeration.

### Part 5: Applied Case Study — Drone Delivery Network & Cruise Speed Optimization (Problem 21)
- **Instance Parameters:** $n = 40 + 2k$ customer demand points in an $8 \times 8\,\text{km}$ region, $m = 12$ candidate charging hubs, demands $d_i$, hub capacities $Q_j$, fixed operating costs $f_j$, distances $D_{ij}$, and aerodynamic constants ($c_1, c_2, E, \kappa, \gamma$).
- **Flight Physics:** Cruise speed $v\,(\text{m/s})$, energy consumption $e(v) = c_1 v^2 + c_2/v^2\,[\text{Wh/km}]$, operating cost $c(v) = \kappa e(v) + \gamma \frac{1000}{v}\,[\text{Rs/km}]$, and round-trip range $R(v) = E/e(v)$. Feasible link condition: $2 D_{ij} \le R(v)$.
- **Sub-tasks:**
  - **21.1 Range-Optimal Cruise Speed ($v_R^*$):** Analytical range maximization, verifying $e''(v_R^*) > 0$, numerical solution on $[5, 40]$ using Golden Section and Cubic Interpolation, and counting feasible $(i, j)$ delivery pairs.
  - **21.2 Formulation & LP Relaxation:** Problem dimension analysis, writing the dual LP, economic interpretation of dual variables in Rs/hour, and complementary slackness validation.
  - **21.3 Branch-and-Bound for Hub Location:** Branching on binary hub open/close decisions $y_j \in \{0, 1\}$ while keeping service allocations $x_{ij}$ continuous. Comparison of 4 strategies:
    1. Depth-first + first-fractional
    2. Depth-first + most-fractional
    3. Best-bound + first-fractional
    4. Best-bound + most-fractional
    Logging nodes explored, LPs solved, execution time, and optimality gap. Validation against `scipy.optimize.milp`.
  - **21.4 Formulation Matters (Strong vs. Aggregated Linking):** Comparing strong linking constraints ($x_{ij} \le y_j$) against aggregated formulation ($\sum_i x_{ij} \le n y_j$). Integrality gap comparison, geometric 1-hub 2-point analysis, and scalability trade-offs for 1000 points.
  - **21.5 Lagrangian Relaxation:** Dualizing demand constraints $\sum_j x_{ij} = 1$ with multipliers $u_i$. Decomposition into $m$ independent continuous knapsack subproblems. Solution via subgradient optimization with step-halving. Bound progression plotting and comparison against strong LP relaxation bound.
  - **21.6 Cost-Optimal Speed Optimization ($v_F^*$):** Evaluating MILP cost curve $F(v)$ on grid $v \in \{8, \dots, 30\}$. Golden section search to $0.1\,\text{m/s}$ tolerance, unimodality analysis, range cutoff discontinuities, and safe search modifications.

---

## Folder Structure

```
Optimisation-Techniques-Assignment/
├── README.md                   # Complete assignment guide, rules, and overview
├── pyproject.toml              # Project configuration and uv dependency specifications
├── uv.lock                     # Deterministic dependency lockfile
├── requirements.txt            # Python dependencies (for reference)
├── main.py                     # Master runner (python main.py <roll>)
├── generate_instance.py        # Problem 21 drone network generator
│
├── problems/                   # Standalone scripts for each problem (1 to 21)
│   ├── problem_01.py           # Single-variable optimum & grid search
│   ├── problem_02.py           # Multivariable unconstrained & Hessian classification
│   ├── problem_03.py           # Equality constraint: Substitution vs Lagrange vs SLSQP
│   ├── problem_04.py           # Lagrange multiplier sensitivity (tank volume)
│   ├── problem_05.py           # KKT conditions, active sets & contour plot
│   ├── problem_06.py           # Graphical LP & vertex identification
│   ├── problem_07.py           # Standard form conversion & basic solutions
│   ├── problem_08.py           # Simplex method implementation (tableau)
│   ├── problem_09.py           # Two-phase simplex method
│   ├── problem_10.py           # Simplex special cases (unbounded, alt optima, infeasible)
│   ├── problem_11.py           # Dual of general LP & complementary slackness
│   ├── problem_12.py           # Shadow prices from final tableau
│   ├── problem_13.py           # Dual simplex method
│   ├── problem_14.py           # Unrestricted & exhaustive line search
│   ├── problem_15.py           # Dichotomous search & interval halving
│   ├── problem_16.py           # Fibonacci vs golden section search
│   ├── problem_17.py           # Quadratic interpolation (Rao's method)
│   ├── problem_18.py           # Cubic interpolation (Rao's method)
│   ├── problem_19.py           # Pure integer program (Branch & Bound)
│   ├── problem_20.py           # 0-1 Knapsack problem (Branch & Bound)
│   └── problem_21.py           # Drone hub location & speed optimization case study
│
├── figures/                    # Generated plots (.png)
├── tables/                     # Generated numerical outputs and tables (.txt / .csv)
├── instances/                  # Generated instance JSON files (instance_<roll>.json)
└── report/                     # LaTeX report deliverable
    └── report.tex              # Main LaTeX report source
```

---

## Tech Stack & Tooling

- **Programming Language:** Python 3.10+ (tested on Python 3.14)
- **Package & Environment Manager:** [`uv`](https://github.com/astral-sh/uv) (Ultra-fast Python package and environment manager)
- **Numerical Computation:** `numpy` (array manipulations, linear algebra, vectorization)
- **Scientific Optimization:** `scipy` (`scipy.optimize.linprog`, `scipy.optimize.minimize`, `scipy.optimize.milp`)
- **Plotting & Visualization:** `matplotlib` (2D curves, 3D contours, search tree diagrams)
- **Report Writing:** LaTeX (`pdflatex` / `xelatex`)

### Library Solvers Policy
Per assignment instructions:
- **Custom implementations are mandatory** for core methods (Tableau Simplex, Two-Phase Simplex, Dual Simplex, Line Search routines, Branch-and-Bound algorithms, Lagrangian subgradient method).
- **Library solvers (`scipy.optimize`) are strictly for cross-checking** and validation.

---

## Rules & How to Run the Project

### 1. Student Parameter Derivation Rules
Extract the last four digits of your roll number: $d_1 d_2 d_3 d_4$ ($d_4 = \text{last digit}$; ignore letters; pad with leading zeros if needed).

$$\begin{aligned}
p &= (d_4 \bmod 5) + 1 \in \{1, \dots, 5\} \\
q &= (d_3 \bmod 4) + 1 \in \{1, \dots, 4\} \\
r &= (d_2 \bmod 3) + 1 \in \{1, 2, 3\} \\
k &= \left((d_1 + d_2 + d_3 + d_4) \bmod 9\right) + 1 \in \{1, \dots, 9\}
\end{aligned}$$

*Example:* Roll `220457` $\implies d_1=0, d_2=4, d_3=5, d_4=7 \implies p=3, q=2, r=2, k=8$.

### 2. Mandatory Parameter Banner
Every script must output the standard parameter table at startup:
```
============================================================
Your roll number: <roll> | p = <p> | q = <q> | r = <r> | k = <k>
============================================================
```
Work performed with incorrect parameters scores zero.

### 3. Setup Environment
```bash
# Clone the repository
git clone <repo-url>
cd Optimisation-Techniques-Assignment

# Set up virtual environment and install dependencies via uv
uv sync

# (Optional) Activate the virtual environment
source .venv/bin/activate
```

> **Tip:** You can execute commands directly with `uv run` without manually activating the virtual environment (e.g., `uv run python main.py <roll_number>`).

### 4. Running the Entire Suite
To run all 21 problems sequentially and regenerate all figures and tables:
```bash
uv run python main.py <roll_number>
# Or if .venv is activated:
python main.py <roll_number>
```
*Example:*
```bash
uv run python main.py 220457
```

### 5. Running Individual Problems
Each problem script is self-contained and accepts the roll number as a CLI argument:
```bash
uv run python problems/problem_01.py <roll_number>
uv run python problems/problem_08.py <roll_number>
uv run python problems/problem_21.py <roll_number>
```

### 6. Generating Problem 21 Instance
Generate the drone delivery network instance JSON:
```bash
python generate_instance.py <roll_number>
```
This produces `instances/instance_<roll>.json` seeded with your full roll number.

### 7. Compiling the LaTeX Report
```bash
cd report
pdflatex report.tex
bibtex report        # if citations used
pdflatex report.tex
pdflatex report.tex
```
