import sys
from problems import (
    problem_01, problem_02, problem_03, problem_04, problem_05,
    problem_06, problem_07, problem_08, problem_09, problem_10,
    problem_11, problem_12, problem_13, problem_14, problem_15,
    problem_16, problem_17, problem_18, problem_19, problem_20,
    problem_21
)

def main():
    roll = sys.argv[1] if len(sys.argv) > 1 else "25027"
    
    modules = [
        problem_01, problem_02, problem_03, problem_04, problem_05,
        problem_06, problem_07, problem_08, problem_09, problem_10,
        problem_11, problem_12, problem_13, problem_14, problem_15,
        problem_16, problem_17, problem_18, problem_19, problem_20,
        problem_21
    ]
    
    print("=" * 65)
    print(f" RUNNING OPTIMIZATION TECHNIQUES ASSIGNMENT SUITE FOR ROLL: {roll}")
    print("=" * 65)
    
    for i, mod in enumerate(modules, 1):
        print("\n" + "=" * 65)
        print(f" PROBLEM {i}")
        print("=" * 65)
        mod.run(roll)
        
    print("\n" + "=" * 65)
    print(" ALL 21 PROBLEMS COMPLETED SUCCESSFULLY.")
    print("=" * 65)

if __name__ == "__main__":
    main()
