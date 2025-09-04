import os, sys, pathlib, pytest
def _in_rag():
    return os.getenv("CONDA_DEFAULT_ENV") == "rag" or "/envs/rag/bin/python" in pathlib.Path(sys.executable).as_posix()
def pytest_sessionstart(session):
    if not _in_rag():
        pytest.exit(f"Tests must run under 'rag'. Found: {sys.executable}", returncode=2)
