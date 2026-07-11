import sys, os
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def path_builder(path):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", path)