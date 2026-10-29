import sys
import os
import importlib

def main():
    roll = sys.argv[1] if len(sys.argv) > 1 else "25027"
    
    os.makedirs("figures", exist_ok=True)
    os.makedirs("tables", exist_ok=True)
    
    print("=" * 65)
    print(f" RUNNING OPTIMIZATION TECHNIQUES ASSIGNMENT SUITE FOR ROLL: {roll}")
    print("=" * 65)
    
    for i in range(1, 22):
        mod_name = f"problems.problem_{i:02d}"
        print("\n" + "=" * 65)
        print(f" PROBLEM {i}")
        print("=" * 65)
        mod = importlib.import_module(mod_name)
        mod.run(roll)
        
    print("\n" + "=" * 65)
    print(" ALL 21 PROBLEMS COMPLETED SUCCESSFULLY.")
    print(" Figures regenerated in figures/")
    print("=" * 65)

if __name__ == "__main__":
    main()
