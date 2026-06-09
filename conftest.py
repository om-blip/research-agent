import sys
import os

# Add the project root to Python path so all modules are importable.
# Without this, pytest running from CI can't find 'rag', 'agents', etc.
# conftest.py in the root directory is automatically loaded by pytest
# before any tests run — making it the right place for path setup.
sys.path.insert(0, os.path.dirname(__file__))