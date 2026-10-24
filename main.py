"""
Master CLI Entrypoint for Applied Optimization Homework.

Usage:
    python main.py <roll_number>
Example:
    python main.py 25027
"""

import sys
from pathlib import Path

# Add src to python path for module imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from optimisation_techniques_assignment.config import compute_student_params


def main() -> None:
    roll_number = sys.argv[1] if len(sys.argv) > 1 else "25027"
    params = compute_student_params(roll_number)
    params.print_header()

    print(f"\nOptimization Techniques Homework Suite initialized for Roll Number: {roll_number}")
    print(f"Parameters: p = {params.p}, q = {params.q}, r = {params.r}, k = {params.k}")
    print(f"Drone Network Instance: n = {params.n} demand points, m = {params.m} hubs\n")
    print("Project structure is set up. Consult README.md for problem walkthroughs and implementation methodology.")


if __name__ == "__main__":
    main()
