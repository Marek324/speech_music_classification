# main.py
# Marek Hric

import sys

if "-h" in sys.argv or "--help" in sys.argv:
    # TODO: implement
    print("Usage: python main.py")
    sys.exit(0)

try:
    import numpy as np
except ImportError as e:
    print(f"Error importing: {e}")
    print("Refer to README.md for installation instructions.")
    sys.exit(1)

def main():
    print(np.array([1, 2, 3]))

if __name__ == "__main__":
    main()
