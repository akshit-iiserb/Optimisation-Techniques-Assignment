# DSE 204: Optimization Techniques — Assignment Submission

**Course Code:** DSE 204  
**Student Name:** Akshit Pandey  
**Roll Number:** 25027  
**Parameters:** $p = 3, \quad q = 3, \quad r = 1, \quad k = 6$

---

## Submission Package Overview

As instructed in the submission guidelines, all materials have been structured into the folder **`DSE204_25027`** and packaged as **`DSE204_25027.zip`**.

```
DSE204_25027/
├── pdf/
│   └── DSE204_25027.pdf      # Complete (H) handwritten solutions for all 21 problems
│
└── codes/                    # Complete (C) computational code and artifacts
    ├── README.md             # Code execution instructions and environment details
    ├── main.py               # Master runner (executes all 21 problems)
    ├── generate_instance.py  # Network generator for Problem 21
    ├── instance_25027.json   # Deterministic instance data for Roll 25027
    ├── output_25027.txt      # Full execution output, tableaus, iteration tables & convergence
    ├── requirements.txt      # Dependency list
    ├── pyproject.toml        # Project configuration
    ├── uv.lock               # Lockfile for exact reproduction
    ├── figures/              # Generated plots and visualization artifacts
    │   ├── problem_01.png
    │   ├── problem_05.png
    │   ├── problem_06.png
    │   ├── problem_16.png
    │   ├── problem_21_cost_speed.png
    │   └── problem_21_lagrangian.png
    └── problems/             # Standalone Python scripts for each problem (01 to 21)
        ├── problem_01.py
        ├── problem_02.py
        ├── ...
        └── problem_21.py
```

---

## Instructions for Running the Code

### 1. Requirements & Dependencies
The computational suite requires Python 3.10+ and the following scientific packages:
* `numpy >= 1.24.0`
* `scipy >= 1.10.0`
* `matplotlib >= 3.7.0`

Install using pip:
```bash
pip install -r DSE204_25027/codes/requirements.txt
```
*(Or with uv: `uv sync`)*

### 2. Running the Complete Suite
Navigate to the `DSE204_25027/codes` directory and run:
```bash
python main.py 25027
```
*(Or with uv: `uv run python main.py 25027`)*

This regenerates all numerical tables, simplex tableaus, iteration sequences, and output figures in the `figures/` folder.

### 3. Running Individual Problems
Each problem script in `problems/` can also be run independently:
```bash
python problems/problem_08.py
python problems/problem_13.py
python problems/problem_21.py
```

---

## Submission Archives
* **Folder:** `DSE204_25027/`
* **ZIP Archive:** `DSE204_25027.zip`
