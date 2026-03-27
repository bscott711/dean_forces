import os
import sys

# Ensure the package is importable
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

try:
    from dean_forces.gui import run_gui
except ImportError:
    # Fallback if running directly from src/dean_forces
    from src.dean_forces.gui import run_gui

if __name__ == "__main__":
    run_gui()
