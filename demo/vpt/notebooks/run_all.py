"""Executes every notebook in this folder in place, so they carry their outputs."""
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

HERE = Path(__file__).parent
for path in sorted(HERE.glob("0*.ipynb")) if len(sys.argv) < 2 else [HERE / a for a in sys.argv[1:]]:
    nb = nbformat.read(path, as_version=4)
    NotebookClient(nb, timeout=3600, kernel_name="python3", resources={"metadata": {"path": str(HERE)}}).execute()
    nbformat.write(nb, path)
    print("executed", path.name, flush=True)
