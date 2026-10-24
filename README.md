# Applied Optimization / Optimization Techniques — Comprehensive Project Guide

> **Course:** Applied Optimization / Optimization Techniques (150 Marks Total)  
> **Student Name:** Akshit  
> **Roll Number:** `25027`  
> **Calculated Parameters:** $p = 3, \quad q = 3, \quad r = 1, \quad k = 6$  
> **Environment:** Python (NumPy, SciPy, Matplotlib) managed via `uv`

---

## Table of Contents
1. [Executive Overview & Honor Code Rules](#1-executive-overview--honor-code-rules)
2. [Roll Number Parameter Derivation (Roll No: 25027)](#2-roll-number-parameter-derivation-roll-no-25027)
3. [Comprehensive Walkthrough & Methodology Plan](#3-comprehensive-walkthrough--methodology-plan)
   - [Part I: Unconstrained & Classical Nonlinear Optimization (Problems 1–5)](#part-i-unconstrained--classical-nonlinear-optimization-problems-15)
   - [Part II: Linear Programming & The Simplex Method (Problems 6–13)](#part-ii-linear-programming--the-simplex-method-problems-613)
   - [Part III: One-Dimensional Line Search Methods (Problems 14–18)](#part-iii-one-dimensional-line-search-methods-problems-1418)
   - [Part IV: Integer Programming & Branch-and-Bound (Problems 19–20)](#part-iv-integer-programming--branch-and-bound-problems-1920)
   - [Part V: Capstone Drone Delivery Network Optimization (Problem 21)](#part-v-capstone-drone-delivery-network-optimization-problem-21)
4. [Project Structure & Architecture](#4-project-structure--architecture)
5. [Tech Stack](#5-tech-stack)
6. [Local Environment Setup & Execution Guide](#6-local-environment-setup--execution-guide)
7. [Deliverables Checklist](#7-deliverables-checklist)

---

## 1. Executive Overview & Honor Code Rules

This repository houses the computational and theoretical framework for the Applied Optimization course homework (150 marks). Every problem features a dual requirement:
1. **(H) Handwritten Component (2 marks each):** Theoretical formulation, analytical derivatives, first few iteration steps, active set tests, or full simplex tableaus with student-specific numbers.
2. **(C) Code Component (3 marks each):** Custom programmatic implementation using **NumPy from scratch**. Library solvers (`scipy.optimize.linprog`, `scipy.optimize.minimize`, `scipy.optimize.milp`) are reserved exclusively for verification and cross-checking.

### Core Integrity & Presentation Requirements
- **Consistency Check:** The numbers computed in the handwritten submission (H) must match the initial iterations and outputs printed by the code (C) to **4 decimal places**.
- **Self-Contained Parameter Extraction:** Every Python script must extract or accept the roll number, compute $p, q, r, k$, print them at the very top, and substitute them dynamically into problem definitions.
- **Reporting:** Every line-search algorithm must print iteration tables detailing intervals/experiment points, function values, interval lengths, and cumulative function evaluations.
- **Unified Runner:** The entire suite must execute via a single CLI invocation:
  ```bash
  python main.py 25027
  ```
  regenerating all benchmark tables and figures.

---

## 2. Roll Number Parameter Derivation (Roll No: 25027)

The assignment mandates student-specific coefficient assignment using the last 4 digits of the roll number.

### 2.1 Extraction Formula
Given roll number $\dots d_1 d_2 d_3 d_4$ where $d_4$ is the last digit:
- $p = (d_4 \bmod 5) + 1$
- $q = (d_3 \bmod 4) + 1$
- $r = (d_2 \bmod 3) + 1$
- $k = ((d_1 + d_2 + d_3 + d_4) \bmod 9) + 1$

### 2.2 Substitution for Roll Number `25027`
For roll number **25027**, the digits are:
$$\text{Full Roll Number: } 25027 \implies d_1 = 5, \quad d_2 = 0, \quad d_3 = 2, \quad d_4 = 7$$

Calculating each parameter:
- **$p$:** $(7 \bmod 5) + 1 = 2 + 1 = \mathbf{3}$
- **$q$:** $(2 \bmod 4) + 1 = 2 + 1 = \mathbf{3}$
- **$r$:** $(0 \bmod 3) + 1 = 0 + 1 = \mathbf{1}$
- **Digit Sum:** $d_1 + d_2 + d_3 + d_4 = 5 + 0 + 2 + 7 = 14$
- **$k$:** $(14 \bmod 9) + 1 = 5 + 1 = \mathbf{6}$

### 2.3 Parameter Summary Table
| Parameter | Formula | Computation | Value for Roll 25027 |
| :--- | :--- | :--- | :---: |
| **$p$** | $(d_4 \bmod 5) + 1$ | $(7 \bmod 5) + 1$ | **3** |
| **$q$** | $(d_3 \bmod 4) + 1$ | $(2 \bmod 4) + 1$ | **3** |
| **$r$** | $(d_2 \bmod 3) + 1$ | $(0 \bmod 3) + 1$ | **1** |
| **$k$** | $(\sum d_i \bmod 9) + 1$ | $(14 \bmod 9) + 1$ | **6** |
| **Drone Points ($n$)** | $40 + 2k$ | $40 + 2(6)$ | **52** |
| **Drone Hubs ($m$)** | Fixed | Fixed | **12** |

---

## 3. Comprehensive Walkthrough & Methodology Plan

---

### Part I: Unconstrained & Classical Nonlinear Optimization (Problems 1–5)

#### Problem 1 — Single-Variable Optimum
- **Problem Statement:**  
  Let $a = p = 3$ and $b = p + q = 3 + 3 = 6$, $r = 1$.  
  $$f(x) = 3x^4 - 4(a + b)x^3 + 6ab x^2 + r = 3x^4 - 36x^3 + 108x^2 + 1$$
- **(H) Methodology:**
  1. Compute first derivative: $f'(x) = 12x^3 - 12(a+b)x^2 + 12ab x = 12x(x-a)(x-b) = 12x(x-3)(x-6)$.
  2. Stationary points: $x \in \{0, 3, 6\}$.
  3. Compute second derivative: $f''(x) = 36x^2 - 72x + 108 = 12(3x^2 - 6x + 9)$.
  4. Evaluate $f''(x)$ at stationary points:
     - At $x = 0$: $f''(0) = 108 > 0 \implies$ Local minimum ($f(0) = 1$).
     - At $x = 3$: $f''(3) = 12(27 - 18 + 9) = 12(-9) = -108 < 0 \implies$ Local maximum ($f(3) = 136$).
     - At $x = 6$: $f''(6) = 108 > 0 \implies$ Local minimum ($f(6) = 1$).
  5. Identify global minimum over $\mathbb{R}$: Compare candidate values $f(0) = 1$ and $f(6) = 1$. Both are global minimizers with value $1$.
- **(C) Methodology:**
  1. Generate a fine grid over $[-1, b + 1] = [-1, 7]$ with $10^5$ points using `np.linspace`.
  2. Vectorized function evaluation and find numerical argmin and min.
  3. Mark stationary points with distinct scatter markers on the Matplotlib plot.

---

#### Problem 2 — Multivariable Unconstrained Optimization
- **Problem Statement:**  
  $$f(x_1, x_2, x_3) = x_1^3 + x_2^3 - 3p^2 x_1 - 3q^2 x_2 + (x_3 - k)^2 + r$$
  For roll 25027: $f(x_1, x_2, x_3) = x_1^3 + x_2^3 - 27x_1 - 27x_2 + (x_3 - 6)^2 + 1$.
- **(H) Methodology:**
  1. Gradient:
     $$\nabla f = \begin{bmatrix} 3x_1^2 - 27 \\ 3x_2^2 - 27 \\ 2(x_3 - 6) \end{bmatrix} = \mathbf{0} \implies x_1 = \pm 3, \quad x_2 = \pm 3, \quad x_3 = 6$$
  2. 4 stationary points: $(\pm 3, \pm 3, 6)$.
  3. Hessian Matrix:
     $$H(x_1, x_2, x_3) = \begin{bmatrix} 6x_1 & 0 & 0 \\ 0 & 6x_2 & 0 \\ 0 & 0 & 2 \end{bmatrix}$$
  4. Principal Minor Test (Sylvester's Criterion):
     - Point $(3, 3, 6)$: $\Delta_1 = 18 > 0, \Delta_2 = 324 > 0, \Delta_3 = 648 > 0 \implies$ Positive Definite $\implies$ **Local Minimum**.
     - Point $(-3, -3, 6)$: $\Delta_1 = -18 < 0, \Delta_2 = 324 > 0, \Delta_3 = -648 < 0 \implies$ Negative Definite $\implies$ **Local Maximum**.
     - Points $(3, -3, 6)$ and $(-3, 3, 6)$: Indefinite $\implies$ **Saddle Points**.
- **(C) Methodology:**
  1. Compute numerical Hessian eigenvalues using `np.linalg.eigvalsh`.
  2. Implement a multi-start runner: launch `scipy.optimize.minimize` (BFGS / Nelder-Mead) from 5 uniformly sampled random starts in $[-6, 6] \times [-6, 6] \times [0, 12]$.
  3. Report basins of attraction and convergence counts.

---

#### Problem 3 — Equality Constraints: Substitution vs. Lagrange Multipliers
- **Problem Statement:**  
  $$\min f(x_1, x_2, x_3) = 3x_1^2 + 3x_2^2 + 1x_3^2 \quad \text{s.t.} \quad x_1 + x_2 + x_3 = 10 + 6 = 16$$
- **(H) Methodology:**
  1. Lagrangian formulation: $\mathcal{L}(x_1, x_2, x_3, \lambda) = 3x_1^2 + 3x_2^2 + x_3^2 - \lambda(x_1 + x_2 + x_3 - 16)$.
  2. 4 necessary conditions:
     $$\frac{\partial \mathcal{L}}{\partial x_1} = 6x_1 - \lambda = 0, \quad \frac{\partial \mathcal{L}}{\partial x_2} = 6x_2 - \lambda = 0, \quad \frac{\partial \mathcal{L}}{\partial x_3} = 2x_3 - \lambda = 0, \quad x_1 + x_2 + x_3 = 16$$
  3. Express variables in terms of $\lambda$: $x_1 = \lambda/6, x_2 = \lambda/6, x_3 = \lambda/2$.
  4. Substitute into constraint to solve for $\lambda^*$ and $x^*$.
  5. Bordered Hessian sufficiency verification:
     $$H_B = \begin{bmatrix} 0 & 1 & 1 & 1 \\ 1 & 6 & 0 & 0 \\ 1 & 0 & 6 & 0 \\ 1 & 0 & 0 & 2 \end{bmatrix}$$
     Check leading principal minors of $H_B$ to confirm strict local minimizer.
- **(C) Methodology:**
  1. Analytical solver function using NumPy.
  2. Substitution method: Eliminate $x_3 = 16 - x_1 - x_2$, formulate unconstrained 2D quadratic $g(x_1, x_2)$, and solve linear system $\nabla g = 0$ via `np.linalg.solve`.
  3. Numerical verification with `scipy.optimize.minimize(method='SLSQP')`.
  4. Confirm all 3 methods agree to 4 decimal places.

---

#### Problem 4 — Lagrange Multiplier as Sensitivity
- **Problem Statement:**  
  Cylindrical tank of radius $x_1$ and height $x_2$ holding volume $V_0 = 50(p+q) = 50(3+3) = 300\text{ m}^3$.
  $$\min S(x_1, x_2) = 2\pi x_1^2 + 2\pi x_1 x_2 \quad \text{s.t.} \quad \pi x_1^2 x_2 = V_0$$
- **(H) Methodology:**
  1. Set up Lagrangian $\mathcal{L} = 2\pi x_1^2 + 2\pi x_1 x_2 - \lambda(\pi x_1^2 x_2 - V_0)$.
  2. Derive first-order conditions showing optimal geometry requires $x_2^* = 2x_1^*$ (height equals diameter).
  3. Solve for $x_1^* = (V_0 / (2\pi))^{1/3}$, $x_2^* = 2x_1^*$, $S^*$, and analytical multiplier $\lambda^* = \frac{2}{x_1^*}$.
  4. Taylor sensitivity approximation: $\Delta S^* \approx \lambda^* \cdot \Delta V_0$ for $\Delta V_0 = 1$.
- **(C) Methodology:**
  1. Solve base problem at $V_0 = 300$ using SLSQP.
  2. Solve perturbed problem at $V_0 + 1 = 301$.
  3. Compute actual $\Delta S_{\text{actual}} = S^*(301) - S^*(300)$ and compare relative error against $\lambda^*$.

---

#### Problem 5 — Karush-Kuhn-Tucker (KKT) Conditions
- **Problem Statement:**  
  $$\min f(x_1, x_2) = (x_1 - 4)^2 + (x_2 - 4)^2 \quad \text{s.t.} \quad x_1 + x_2 \le 1, \quad x_1 \ge 0, \quad x_2 \ge 0$$
- **(H) Methodology:**
  1. Convexity proof: Objective is strictly convex ($\nabla^2 f = 2I \succ 0$) and constraint set is an affine simplex (convex polytope). Thus, KKT conditions are both necessary and sufficient for global optimality.
  2. KKT System:
     - Stationarity: $\begin{bmatrix} 2(x_1 - 4) \\ 2(x_2 - 4) \end{bmatrix} + \mu_1 \begin{bmatrix} 1 \\ 1 \end{bmatrix} - \mu_2 \begin{bmatrix} 1 \\ 0 \end{bmatrix} - \mu_3 \begin{bmatrix} 0 \\ 1 \end{bmatrix} = \mathbf{0}$
     - Primal feasibility: $x_1 + x_2 \le 1, x_1 \ge 0, x_2 \ge 0$.
     - Dual feasibility: $\mu_1, \mu_2, \mu_3 \ge 0$.
     - Complementary slackness: $\mu_1(x_1 + x_2 - 1) = 0, \mu_2 x_1 = 0, \mu_3 x_2 = 0$.
  3. Active set exploration: Test all combinations of active constraints. Show that only the active set $\{x_1 + x_2 = 1\}$ yields strictly positive multipliers $\mu_1 > 0, \mu_2 = 0, \mu_3 = 0$.
- **(C) Methodology:**
  1. Solve via SLSQP.
  2. Recover multipliers directly from the stationarity linear equation: $\mu = - (A_{\text{active}}^T)^{-1} \nabla f(x^*)$.
  3. Generate 2D contour plot depicting objective concentric circles, constraint boundary line, feasible region shading, and normal gradient vectors showing KKT balance.

---

### Part II: Linear Programming & The Simplex Method (Problems 6–13)

#### Problem 6 — Graphical Simplex Method
- **Problem Statement:**  
  $$\max z = 5x_1 + 4x_2 \quad \text{s.t.} \quad x_1 + 2x_2 \le 11, \quad 3x_1 + x_2 \le 18, \quad x_1 \le 7, \quad x_1, x_2 \ge 0$$
- **(H) Methodology:**
  1. Draw bounding lines on Cartesian plane.
  2. Compute vertices of the feasible polygon.
  3. Evaluate $z(x_1, x_2)$ at every extreme point. Identify optimal vertex and binding constraints.
- **(C) Methodology:**
  1. Implement combinatorial line-intersection routine for all pairs of constraints $\binom{5}{2} = 10$.
  2. Filter intersections through inequality matrix to extract feasible extreme points.
  3. Solve with `scipy.optimize.linprog(method='highs')` and plot the feasible region polygon with level curves of $z$.

---

#### Problem 7 — Standard Form & Basic Solutions
- **Problem Statement:**  
  - Part (a): Convert to standard form (all variables $\ge 0$, equality constraints, RHS $\ge 0$):  
    $$\min z = 3x_1 - 3x_2 + 3x_3 \quad \text{s.t.} \quad x_1 + x_2 - x_3 \ge 1, \quad 2x_1 - x_2 + x_3 \le 26, \quad x_1 + x_2 + x_3 = 9$$
    with $x_1 \ge 0, x_2 \le 0, x_3$ unrestricted.
  - Part (b): For $\max z = 4x_1 + 4x_2$ s.t. $2x_1 + x_2 \le 9, x_1 + 3x_2 \le 15, x \ge 0$, add slacks $s_1, s_2$.
- **(H) Methodology:**
  1. Substitutions: $x_2 = -x_2'$ ($x_2' \ge 0$), $x_3 = x_3^+ - x_3^-$ ($x_3^+, x_3^- \ge 0$). Introduce surplus $e_1 \ge 0$ and slack $s_2 \ge 0$.
  2. For (b), write equality system $A x = b$ ($A \in \mathbb{R}^{2 \times 4}$). Hand-calculate 3 of $\binom{4}{2} = 6$ basic solutions and classify feasibility.
- **(C) Methodology:**
  1. Solve both original and standard form formulations with `linprog` to verify matching objective value.
  2. Enumerate all 6 bases $\binom{4}{2}$, solve $x_B = B^{-1}b$, check $x_B \ge 0$, and pick the best BFS.

---

#### Problem 8 — Custom Simplex Tableau Implementation
- **Problem Statement:**  
  $$\max z = 5x_1 + 6x_2 + 2x_3 \quad \text{s.t.} \quad x_1 + 2x_2 + x_3 \le 16, \quad 2x_1 + x_2 + 2x_3 \le 15, \quad x_1 + x_2 + 3x_3 \le 21, \quad \mathbf{x} \ge \mathbf{0}$$
- **(H) Methodology:**
  1. Set up initial tableau with slacks $s_1, s_2, s_3$.
  2. Identify entering variable (most positive in $z$-row or most negative reduced cost).
  3. Compute minimum ratio test $\theta = \min_{i: a_{ij} > 0} \{b_i / a_{ij}\}$ to identify leaving variable.
  4. Perform Gauss-Jordan pivot operations. Record final optimal tableau for reuse in Problem 12.
- **(C) Methodology:**
  1. Write a pure NumPy `TableauSimplex` class from scratch:
     - Initialization, pivot selection (Bland's rule / Dantzig rule), row operations.
     - Formatted string representation printing the complete tableau at every iteration.
  2. Cross-check final $x^*$ and $z^*$ with `scipy.optimize.linprog`.

---

#### Problem 9 — Two-Phase Simplex Method
- **Problem Statement:**  
  $$\min z = 4x_1 + 5x_2 \quad \text{s.t.} \quad x_1 + x_2 \ge 5, \quad x_1 + 3x_2 \ge 9, \quad x_1 + x_2 \le 18, \quad \mathbf{x} \ge \mathbf{0}$$
- **(H) Methodology:**
  1. Add surplus variables $e_1, e_2$, slack $s_3$, and artificial variables $a_1, a_2$.
  2. Phase I objective: $\min W = a_1 + a_2$. Canonicalize initial tableau by subtracting rows containing artificial variables from the objective row.
  3. Perform first two Phase I pivots manually.
- **(C) Methodology:**
  1. Extend `TableauSimplex` with an automated Two-Phase driver:
     - Phase I setup, pivoting until $W^* = 0$.
     - Column dropping / zeroing of artificial variables.
     - Phase II restoration of original objective and pivoting to optimality.
  2. Print Phase I ending tableau and Phase II final tableau. Cross-check with `linprog`.

---

#### Problem 10 — Special Cases in Linear Programming
- **Test Instances:**  
  - (a) Unbounded: $\max z = 3x_1 + 3x_2$ s.t. $-x_1 + x_2 \le 1, x_1 - 2x_2 \le 3, x \ge 0$.
  - (b) Alternative Optima: $\max z = 1x_1 + 2x_2$ s.t. $x_1 + 2x_2 \le 13, x_1 \le 6, x_2 \le 8, x \ge 0$.
  - (c) Infeasible: $\max z = x_1 + x_2$ s.t. $2x_1 + x_2 \le 5, x_1 + 2x_2 \le 5, x_1 + x_2 \ge 6, x \ge 0$.
- **(H) Methodology:**
  1. For (a): Show that entering variable has a pivot column with all entries $\le 0$ (unbounded ray).
  2. For (b): Identify a non-basic variable with zero reduced cost in optimal tableau. Pivot to find the second optimal vertex and write parametric segment $x(\alpha) = \alpha x^{(1)} + (1-\alpha) x^{(2)}, \alpha \in [0, 1]$.
  3. For (c): Demonstrate Phase I termination with optimal $W > 0$ and artificial variable in basis.
- **(C) Methodology:**
  1. Add automated status detection flags to custom simplex code (`OPTIMAL`, `UNBOUNDED`, `INFEASIBLE`, `ALTERNATIVE_OPTIMA`).
  2. Compare against `linprog` status codes (e.g., status 2 for infeasible, status 3 for unbounded).

---

#### Problem 11 — Writing the Dual of a General LP
- **Problem Statement:**  
  $$\max z = 5x_1 + 4x_2 + 4x_3 \quad \text{s.t.} \quad x_1 + x_2 + x_3 \le 16, \quad 2x_1 - x_2 + x_3 \ge 1, \quad x_1 + 2x_2 - x_3 = 7$$
  with $x_1, x_2 \ge 0, x_3$ unrestricted.
- **(H) Methodology:**
  1. Construct dual systematically via primal-dual mapping table:
     - Constraint 1 ($\le$) $\implies y_1 \ge 0$.
     - Constraint 2 ($\ge$) $\implies y_2 \le 0$.
     - Constraint 3 ($=$) $\implies y_3 \in \mathbb{R}$ (unrestricted).
     - Primal variable $x_3$ unrestricted $\implies$ dual constraint 3 is an equality ($=$).
  2. State one-line justification for every variable sign restriction and constraint direction.
- **(C) Methodology:**
  1. Formulate and solve primal and dual models using `scipy.optimize.linprog`.
  2. Confirm strong duality ($z^* = w^*$).
  3. Programmatically evaluate and verify complementary slackness products:
     $$y_i^* (A_i x^* - b_i) = 0 \quad \forall i, \qquad x_j^* (A_j^T y^* - c_j) = 0 \quad \forall j$$

---

#### Problem 12 — Shadow Prices from the Final Tableau
- **Problem Statement:** Re-examine Problem 8 optimal tableau.
- **(H) Methodology:**
  1. Read dual variables $y^*$ directly from slack variable coefficients in row 0.
  2. Verify $w^* = \mathbf{b}^T \mathbf{y}^* = z^*$.
  3. Check complementary slackness against primal basic solution.
- **(C) Methodology:**
  1. Solve dual with `linprog`, verify equivalence to tableau multipliers.
  2. Perturb RHS $b_1 \to b_1 + 1$ ($16 \to 17$), resolve primal, and verify $\Delta z^* = y_1^*$.
  3. Document concise economic explanation: Why a non-binding constraint has a shadow price of zero (slack resource holds zero marginal utility).

---

#### Problem 13 — Dual Simplex Method
- **Problem Statement:**  
  $$\min z = 4x_1 + 4x_2 + 2x_3 \quad \text{s.t.} \quad x_1 + 2x_2 + x_3 \ge 10, \quad 2x_1 + x_2 + 3x_3 \ge 15, \quad x_1 + x_2 + 2x_3 \ge 7, \quad \mathbf{x} \ge \mathbf{0}$$
- **(H) Methodology:**
  1. Multiply $\ge$ rows by $-1$, add slacks $s_1, s_2, s_3$.
  2. Verify initial tableau is dual feasible ($c_j - z_j \ge 0$) but primal infeasible ($b_i < 0$).
  3. Dual Simplex pivot rules:
     - Leaving variable: Row $r$ with most negative RHS $b_r = \min_i \{b_i\}$.
     - Entering variable: Column $s$ selected via ratio test $\min_{j: a_{rj} < 0} \{ (c_j - z_j) / |a_{rj}| \}$.
  4. Perform first two pivots manually.
- **(C) Methodology:**
  1. Implement custom `DualSimplex` tableau solver in NumPy.
  2. Print tableaus at each step; confirm convergence in 2–4 iterations and verify against `linprog`.

---

### Part III: One-Dimensional Line Search Methods (Problems 14–18)

*Mandatory requirement for this part: The code must print an iteration table (interval/points, $f$ values, interval length) and total function evaluations.*

#### Problem 14 — Unrestricted Search & Exhaustive Search
- **Problem Statement:**  
  $$f(\lambda) = 2\lambda^2 + \frac{40}{\lambda}, \quad \lambda > 0$$
- **(H) Methodology:**
  1. Accelerated step search: Start $\lambda_0 = 0.5$, initial step $s = 0.1$, double step $s_{k+1} = 2s_k$ at each iteration. Tabulate $\lambda$ and $f(\lambda)$ until $f$ increases; record bracketing interval $[\lambda_{k-1}, \lambda_{k+1}]$.
  2. Exhaustive search on $[0.5, 5.0]$ with $n = 8$ interior points ($\Delta x = (5.0 - 0.5)/(8+1) = 0.5$). State final bracket.
- **(C) Methodology:**
  1. Custom implementation of accelerated step bracketing.
  2. Exhaustive search implementation. Determine analytical and programmatic minimum $n$ achieving final interval $\le 1\%$ of original ($L_n = \frac{2}{n+1} L_0 \le 0.01 L_0 \implies n \ge 199$).
  3. Compare numerical minimizer against analytical root: $f'(\lambda) = 4\lambda - 40/\lambda^2 = 0 \implies \lambda^* = 10^{1/3} \approx 2.1544$.

---

#### Problem 15 — Dichotomous Search vs. Interval Halving
- **Problem Statement:**  
  Constant $c = 3 + q + 0.1k = 3 + 3 + 0.6 = 6.6$.  
  $$f(\lambda) = e^\lambda - 6.6\lambda, \quad \lambda \in [0, 3]$$
- **(H) Methodology:**
  1. Dichotomous Search: Perturbation $\delta = 0.001$. Two iterations evaluating $\lambda_1 = \frac{a+b-\delta}{2}, \lambda_2 = \frac{a+b+\delta}{2}$, discarding the inferior sub-interval.
  2. Interval Halving: Two iterations evaluating points at $1/4, 2/4, 3/4$ of interval.
- **(C) Methodology:**
  1. Implement both algorithms until interval length $\le 0.01$.
  2. Print iteration tables, count function evaluations, and compute absolute error against exact analytical minimizer $\lambda^* = \ln(6.6) \approx 1.8871$.

---

#### Problem 16 — Fibonacci Search vs. Golden Section Method
- **Problem Statement:**  
  Constant $c = r + 1 + 0.1k = 1 + 1 + 0.6 = 2.6$.  
  $$f(\lambda) = \lambda^4 - 10.4\lambda, \quad \lambda \in [0, 3]$$
- **(H) Methodology:**
  1. Fibonacci Search ($n = 11$, $F_0 = F_1 = 1, \dots, F_{11} = 144$): Initial placement $L_2^* = \frac{F_9}{F_{11}} L_0 = \frac{55}{144} (3)$. Carry out first 3 interval reductions.
  2. Golden Section Method: Step ratio $\gamma = \frac{\sqrt{5}-1}{2} \approx 0.618034$. Perform first 3 reductions.
- **(C) Methodology:**
  1. Implement both algorithms with $n = 11$ evaluations.
  2. Generate a comparative plot of interval length vs. evaluation index on a semilog scale.
  3. Report final intervals and theoretical vs. empirical reduction ratio $L_n / L_0$.

---

#### Problem 17 — Quadratic Interpolation
- **Problem Statement:**  
  $$f(\lambda) = \lambda^4 - 6\lambda^2 - 5\lambda + 5$$
- **(H) Methodology:**
  1. Rao's bracketing: Start $A = 0$, step $t_0 = 0.5$, doubling until $f$ increases to obtain 3 points $A, B, C$.
  2. Fit polynomial $h(\lambda) = a + b\lambda + c\lambda^2$ through $(A, f_A), (B, f_B), (C, f_C)$.
  3. Stationary point of quadratic: $\tilde{\lambda}^* = -b / (2c)$.
  4. Evaluate stopping criterion:
     $$\left| \frac{h(\tilde{\lambda}^*) - f(\tilde{\lambda}^*)}{f(\tilde{\lambda}^*)} \right| \le 0.001$$
     Perform one refit if condition fails.
- **(C) Methodology:**
  1. Implement iterative quadratic interpolation with bracket preservation (retaining best point and 2 adjacent neighbors).
  2. Print parameters of every quadratic fit.
  3. Compare with `scipy.optimize.minimize_scalar` and comment on the looseness of the relative function-value test.

---

#### Problem 18 — Cubic Interpolation
- **Problem Statement:**  
  Constant $c = 10 + 2r + k = 10 + 2(1) + 6 = 18$.  
  $$f(\lambda) = \lambda^5 - 5\lambda^3 - 18\lambda + 5, \quad f'(0) = -18 < 0$$
- **(H) Methodology:**
  1. Bracketing with $t_0 = 0.4$, doubling until $f'(B) > 0$. $A$ is the last point with $f'(A) < 0$.
  2. Apply Rao's cubic interpolation formula:
     $$Z = \frac{3(f_A - f_B)}{B - A} + f'_A + f'_B, \quad Q = \sqrt{Z^2 - f'_A f'_B}, \quad \tilde{\lambda}^* = A + \frac{f'_A + Z \pm Q}{f'_A + f'_B + 2Z}(B - A)$$
     Choose sign ensuring $\tilde{\lambda}^* \in [A, B]$.
  3. Evaluate $f'(\tilde{\lambda}^*)$ and identify which bracket endpoint it replaces.
- **(C) Methodology:**
  1. Implement iterative cubic search terminating when $|f'(\tilde{\lambda}^*)| \le 0.01$.
  2. Compare convergence rate and evaluations against Problem 17 and against the exact root of $f'(\lambda) = 5\lambda^4 - 15\lambda^2 - 18 = 0$.

---

### Part IV: Integer Programming & Branch-and-Bound (Problems 19–20)

#### Problem 19 — Pure Integer Linear Programming (B&B)
- **Problem Statement:**  
  $$\max z = 6x_1 + 5x_2 \quad \text{s.t.} \quad 5x_1 + 4x_2 \le 28, \quad 2x_1 + 5x_2 \le 15, \quad x_1, x_2 \ge 0, \quad x_1, x_2 \in \mathbb{Z}$$
- **(H) Methodology:**
  1. Solve root LP relaxation graphically.
  2. Identify fractional variable with largest fractionality; branch into $x_i \le \lfloor x_i^* \rfloor$ and $x_i \ge \lceil x_i^* \rceil$.
  3. Solve child nodes graphically. Draw branch-and-bound tree to depth 2, labelling each node with LP coordinates, objective $z$, and status (`Branched`, `Integer Feasible`, `Infeasible`, `Fathomed by Bound`).
- **(C) Methodology:**
  1. Implement recursive Depth-First Branch and Bound using `scipy.optimize.linprog(method='highs')` at each node.
  2. Branching strategy: First fractional variable, exploring $\le$ branch first.
  3. ASCII tree printer logging node index, parent, bound, and status.
  4. Brute-force verification over all integer lattice points in the bounding box.

---

#### Problem 20 — 0–1 Knapsack by Branch-and-Bound
- **Problem Statement:**  
  $$\max \sum_{i=1}^6 v_i x_i \quad \text{s.t.} \quad \sum_{i=1}^6 w_i x_i \le 14, \quad x_i \in \{0, 1\}$$
  - Values: $\mathbf{v} = (10+3, 13+3, 7+1, 8+6, 11, 9) = (13, 16, 8, 14, 11, 9)$
  - Weights: $\mathbf{w} = (4, 6, 3, 5, 5, 4)$
  - Capacity: $W = 13 + r = 14$
- **(H) Methodology:**
  1. Compute value-to-weight efficiency ratios:  
     $v/w = [3.25, 2.667, 2.667, 2.800, 2.200, 2.250]$.  
     Sorted order of items: Item 1 ($3.25$), Item 4 ($2.80$), Item 2 ($2.667$), Item 3 ($2.667$), Item 6 ($2.25$), Item 5 ($2.20$).
  2. Solve root continuous knapsack relaxation greedily: Pack items 1, 4, and take fractional part of item 2 or 3.
  3. Branch on critical fractional item ($x_c = 0$ vs $x_c = 1$). Draw B&B tree to depth 2 showing greedy upper bounds.
- **(C) Methodology:**
  1. Adapt Problem 19 B&B solver with variable bounds $0 \le x_i \le 1$.
  2. Compare total nodes explored against full brute-force enumeration ($2^6 = 64$ combinations).

---

### Part V: Capstone Drone Delivery Network Optimization (Problem 21)

*Total Weight: 50 Marks (6 Sub-tasks: 21.1 to 21.6)*  
*Instance Parameters for Roll 25027:* $n = 40 + 2(6) = 52$ demand points, $m = 12$ hubs.

#### 21.1 Range-Optimal Cruise Speed (6 marks)
- **Model Physics:**
  $$e(v) = c_1 v^2 + \frac{c_2}{v^2} \quad [\text{Wh/km}], \quad c(v) = \kappa e(v) + \frac{1000 \gamma}{v} \quad [\text{Rs/km}]$$
  Round-trip range: $R(v) = \frac{E}{e(v)}$. Feasibility condition: $2 D_{ij} \le R(v)$.
- **(H) Methodology:**
  Maximize range $R(v) \iff$ minimize energy consumption $e(v)$.  
  $e'(v) = 2c_1 v - 2c_2 v^{-3} = 0 \implies v_R^* = \left( \frac{c_2}{c_1} \right)^{1/4}$.  
  Verify strict convexity: $e''(v_R^*) = 2c_1 + 6c_2 (v_R^*)^{-4} > 0$.
- **(C) Methodology:**
  1. Solve $\min_{v \in [5, 40]} e(v)$ using Golden Section (Prob 16) and Cubic Interpolation (Prob 18).
  2. Compare function evaluations. Compute maximum range $R(v_R^*)$ and count allowable $(i, j)$ delivery pairs where $2 D_{ij} \le R(v_R^*)$.

---

#### 21.2 Formulation and the LP Relaxation (8 marks)
- **MILP Model (fixed $v = v_R^*$):**
  $$\min \sum_{j=1}^m f_j y_j + \sum_{i=1}^n \sum_{j=1}^m 2 D_{ij} c(v) d_i x_{ij}$$
  $$\text{s.t.} \quad \sum_{j=1}^m x_{ij} = 1 \; (\forall i), \quad \sum_{i=1}^n d_i x_{ij} \le Q_j y_j \; (\forall j), \quad x_{ij} \le y_j \; (\forall i, j), \quad x_{ij} = 0 \text{ if } 2D_{ij} > R(v)$$
- **(H) Methodology:**
  1. Variable & constraint count derivation for $n = 52, m = 12$.
  2. Derive the dual of the LP relaxation.
  3. Provide rigorous economic interpretations for dual multipliers of "every point served" ($\alpha_i$) and "hub capacity" ($\beta_j$) in units of Rs/hour.
- **(C) Methodology:**
  1. Assemble sparse constraint matrices using SciPy/NumPy and solve LP relaxation using `scipy.optimize.linprog(method='highs')`.
  2. Extract marginals (dual values). Identify the most expensive demand point and explain why via transport distance and demand rate.
  3. Verify complementary slackness for capacity constraints: $\beta_j^* \left( Q_j y_j^* - \sum_i d_i x_{ij}^* \right) = 0$.

---

#### 21.3 Branch-and-Bound Variants for Hub Selection (10 marks)
- **(C) Methodology:**
  Extend custom B&B algorithm to branch **only on binary hub decisions $y_j \in \{0, 1\}$** while keeping $x_{ij}$ continuous. Implement and benchmark 4 exploration strategies:
  1. **DFS + First-Fractional**
  2. **DFS + Most-Fractional**
  3. **Best-Bound (A*) + First-Fractional**
  4. **Best-Bound (A*) + Most-Fractional**
  Log and tabulate: Total nodes explored, LPs solved, execution time, and final optimality gap. Verify global optimum against `scipy.optimize.milp`.
- **(H) Methodology:**
  Draw top two levels of the search tree for the best variant and document one fathoming decision (by bound, integrality, or infeasibility).

---

#### 21.4 Formulation Matters: Strong vs. Aggregated Linking (8 marks)
- **Formulation Comparison:**
  - Strong Formulation: $x_{ij} \le y_j \quad \forall i, j$ ($n \times m = 624$ constraints).
  - Weak / Aggregated Formulation: $\sum_{i=1}^n x_{ij} \le n y_j \quad \forall j$ ($m = 12$ constraints).
- **(C) Methodology:**
  Solve LP relaxation and B&B under both formulations. Tabulate LP lower bound $z_{LP}$, integrality gap $\frac{z_{IP} - z_{LP}}{z_{IP}}$, and total node count.
- **(H) Methodology:**
  Provide geometric proof using a 1-hub, 2-point toy example illustrating that strong linking produces a tighter polyhedral projection. Discuss scalability trade-offs for large networks ($n = 1000$).

---

#### 21.5 Lagrangian Relaxation & Subgradient Optimization (10 marks)
- **(H) Methodology:**
  Dualize the complicating demand satisfaction constraints $\sum_j x_{ij} = 1$ with multipliers $u_i \in \mathbb{R}$:
  $$L(\mathbf{u}) = \sum_{i=1}^n u_i + \min_{y, x} \left\{ \sum_{j=1}^m f_j y_j + \sum_{i=1}^n \sum_{j=1}^m \left( 2 D_{ij} c(v) d_i - u_i \right) x_{ij} \right\}$$
  Show that this decomposes into $m$ independent continuous knapsack subproblems (one per hub $j$), each solvable greedily in $O(n \log n)$. Prove that $L(\mathbf{u}) \le z_{IP} \; \forall \mathbf{u}$.
- **(C) Methodology:**
  1. Implement Polyak subgradient optimization:
     $$t_k = \theta_k \frac{z_{UB} - L(\mathbf{u}^k)}{\| \mathbf{g}^k \|^2}, \quad g_i^k = 1 - \sum_j x_{ij}^k$$
     Halve $\theta$ after 20 consecutive non-improving steps.
  2. Primal heuristic: At each iteration, open the hubs selected by the Lagrangian subproblems ($y_j = 1$) and solve the reduced assignment LP to generate valid upper bound $z_{UB}$.
  3. Plot lower bound $L(\mathbf{u}^k)$ and upper bound $z_{UB}^k$ across iterations.
  4. Compare best Lagrangian bound against strong LP bound and explain the Integrality Property result.

---

#### 21.6 Cost-Optimal Drone Cruise Speed (8 marks)
- **(C) Methodology:**
  1. Define $F(v)$ as optimal MILP total cost at cruise speed $v$.
  2. Compute $F(v)$ over grid $v \in \{8, 9, \dots, 30\}$ m/s and plot the curve.
  3. Minimize $F(v)$ via Golden Section search on $[8, 30]$ to tolerance $0.1$ m/s. Report $v_F^*$ and compare with range-optimal speed $v_R^*$.
- **(H) Methodology:**
  Theoretical analysis: Analyze whether $F(v)$ is unimodal or discontinuous. Explain why discrete jumps occur (due to range cutoff $2 D_{ij} \le R(v)$ disabling links). Discuss implications for Golden Section search and how to safeguard optimization (e.g. multi-interval or grid pre-screening).

---

## 4. Project Structure & Architecture

To maintain clarity and modularity across all 21 problems, the repository is structured as follows:

```
Optimisation-Techniques-Assignment/
├── README.md                           # Comprehensive documentation & walkthrough
├── pyproject.toml                      # Project metadata & dependencies
├── uv.lock                             # Locked reproducible dependencies
├── main.py                             # Unified CLI runner: python main.py <roll>
├── generate_instance.py                # Instance generator script for Problem 21
├── instances/                          # JSON problem instances
│   └── instance_25027.json             # Seeded instance for Roll 25027
│
├── src/
│   └── optimisation_techniques_assignment/
│       ├── __init__.py
│       ├── config.py                   # Parameter derivation (p, q, r, k)
│       ├── utils.py                    # Formatted logging, table printing, plotting utilities
│       │
│       ├── unconstrained/              # Problems 1 & 2
│       │   ├── prob01_single_var.py
│       │   └── prob02_multivar.py
│       │
│       ├── constrained_nlp/            # Problems 3, 4, 5
│       │   ├── prob03_equality_lagrange.py
│       │   ├── prob04_sensitivity.py
│       │   └── prob05_kkt.py
│       │
│       ├── linear_programming/         # Problems 6 through 13
│       │   ├── prob06_graphical.py
│       │   ├── prob07_standard_form.py
│       │   ├── prob08_simplex_tableau.py
│       │   ├── prob09_twophase_simplex.py
│       │   ├── prob10_special_cases.py
│       │   ├── prob11_duality.py
│       │   ├── prob12_shadow_prices.py
│       │   └── prob13_dual_simplex.py
│       │
│       ├── line_search/                # Problems 14 through 18
│       │   ├── prob14_unrestricted_exhaustive.py
│       │   ├── prob15_dichotomous_halving.py
│       │   ├── prob16_fibonacci_golden.py
│       │   ├── prob17_quadratic_interp.py
│       │   └── prob18_cubic_interp.py
│       │
│       ├── integer_programming/        # Problems 19 & 20
│       │   ├── prob19_branch_and_bound.py
│       │   └── prob20_knapsack_bb.py
│       │
│       └── drone_project/              # Problem 21 (Tasks 21.1 to 21.6)
│           ├── drone_model.py          # Physics (energy, cost, range)
│           ├── prob21_1_range_speed.py
│           ├── prob21_2_lp_relaxation.py
│           ├── prob21_3_bnb_variants.py
│           ├── prob21_4_formulation_comparison.py
│           ├── prob21_5_lagrangian_relaxation.py
│           └── prob21_6_cost_optimal_speed.py
│
├── figures/                            # Output figures generated by scripts
│   ├── prob01_poly_fit.png
│   ├── prob05_kkt_contours.png
│   ├── prob06_feasible_region.png
│   ├── prob16_convergence.png
│   ├── prob21_bounds_evolution.png
│   └── prob21_speed_vs_cost.png
│
├── tables/                             # Output ASCII / LaTeX formatted tables
│   ├── prob14_iterations.txt
│   ├── prob15_iterations.txt
│   ├── prob16_iterations.txt
│   ├── prob21_3_bnb_comparison.tex
│   └── prob21_4_formulation.tex
│
└── report/                             # LaTeX submission document
    ├── report.tex
    └── references.bib
```

---

## 5. Tech Stack

- **Core Programming Language:** Python 3.12+ (tested on Python 3.14)
- **Primary Numerical Engine:** [NumPy](https://numpy.org/) (Custom algorithms, vectorization, matrix algebra)
- **Solver & Benchmark Suite:** [SciPy](https://scipy.org/) (`scipy.optimize.minimize`, `scipy.optimize.linprog`, `scipy.optimize.milp` for cross-checking)
- **Visualization:** [Matplotlib](https://matplotlib.org/) (High-resolution contour plots, convergence rates, and polyhedral graphics)
- **Project & Environment Manager:** [`uv`](https://github.com/astral-sh/uv) (Ultra-fast virtual environment and package dependency resolver)
- **Reporting:** LaTeX / Overleaf for the final formatted document.

---

## 6. Local Environment Setup & Execution Guide

### 6.1 Prerequisites
Ensure you have `git` and `uv` installed on your machine.
If `uv` is not installed, install it via:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 6.2 Setting Up the Environment
Clone or navigate to the project directory:
```bash
cd /home/forest/Desktop/Optimisation-Techniques-Assignment
```

Create and synchronize the virtual environment with all required dependencies:
```bash
uv sync
```
This automatically sets up Python, installs `numpy`, `scipy`, and `matplotlib`, and configures the virtual environment under `.venv`.

### 6.3 Activating the Environment
```bash
# On Linux / macOS:
source .venv/bin/activate
```

### 6.4 Running Individual Modules
You can execute and test any individual problem module:
```bash
# Run Problem 1 (Single-variable unconstrained):
python -m optimisation_techniques_assignment.unconstrained.prob01_single_var 25027

# Run Problem 8 (Custom Simplex Tableau):
python -m optimisation_techniques_assignment.linear_programming.prob08_simplex_tableau 25027

# Run Problem 21.1 (Range-optimal speed):
python -m optimisation_techniques_assignment.drone_project.prob21_1_range_speed 25027
```

### 6.5 Running the Master Pipeline
To run the full suite for your roll number, regenerate all figures and tables, and print all verification reports:
```bash
python main.py 25027
```

---

## 7. Deliverables Checklist

- [ ] **Handwritten Report (H):**
  - Parameter table placed at top of page 1 ($p=3, q=3, r=1, k=6$).
  - Analytical derivations, bordered Hessians, KKT tests, and initial simplex tableaus.
  - Verification that handwritten intermediate numbers match code printouts to 4 decimals.
- [ ] **Computational Repository (C):**
  - NumPy implementation of all simplex variants, 1-D line searches, B&B, and Lagrangian relaxation.
  - Verified outputs matching SciPy / HiGHS cross-checks.
  - Regenerable execution via `python main.py 25027`.
- [ ] **LaTeX Report:** Complete formatted report matching academic standards.
