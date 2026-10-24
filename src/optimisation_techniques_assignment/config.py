"""
Configuration and parameter derivation for Applied Optimization Homework.

Extracts student-specific coefficients based on roll number as specified in the assignment:
- Last four digits: d1 d2 d3 d4 (d4 = last digit)
- p = (d4 mod 5) + 1
- q = (d3 mod 4) + 1
- r = (d2 mod 3) + 1
- k = ((d1 + d2 + d3 + d4) mod 9) + 1
"""

from dataclasses import dataclass
import re
import sys


@dataclass(frozen=True)
class StudentParams:
    roll_number: str
    d1: int
    d2: int
    d3: int
    d4: int
    p: int
    q: int
    r: int
    k: int
    n: int  # Number of demand points for Problem 21: 40 + 2k
    m: int = 12  # Number of candidate hubs for Problem 21

    def print_header(self) -> None:
        """Print the required parameters table mandated by the assignment."""
        print("=" * 60)
        print(f"Your roll number: {self.roll_number} | p = {self.p} | q = {self.q} | r = {self.r} | k = {self.k}")
        print("=" * 60)


def compute_student_params(roll: str | int = "25027") -> StudentParams:
    """Compute and validate student-specific parameters from roll number."""
    roll_str = str(roll).strip()
    digits = re.findall(r"\d", roll_str)

    if len(digits) < 4:
        # Pad with leading zeros if fewer than 4 digits
        digits = ["0"] * (4 - len(digits)) + digits

    d1, d2, d3, d4 = map(int, digits[-4:])

    p = (d4 % 5) + 1
    q = (d3 % 4) + 1
    r = (d2 % 3) + 1
    k = ((d1 + d2 + d3 + d4) % 9) + 1
    n = 40 + 2 * k

    return StudentParams(
        roll_number=roll_str,
        d1=d1,
        d2=d2,
        d3=d3,
        d4=d4,
        p=p,
        q=q,
        r=r,
        k=k,
        n=n,
    )


if __name__ == "__main__":
    roll = sys.argv[1] if len(sys.argv) > 1 else "25027"
    params = compute_student_params(roll)
    params.print_header()
