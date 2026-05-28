"""HuggingFace Spaces entry point."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from loanlens.dashboard.app import demo

if __name__ == "__main__":
    demo.launch()
