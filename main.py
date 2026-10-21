"""
Run gasim from a source checkout without installing it, e.g.

    python main.py run configs/legacy/test_room.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from gasim.cli import main

if __name__ == '__main__':
    sys.exit(main())
