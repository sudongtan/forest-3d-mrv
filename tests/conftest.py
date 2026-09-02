import os

# pycolmap and torch each bundle their own copy of libomp.dylib on macOS; loading both
# in the same process aborts with "OMP: Error #15" (duplicate OpenMP runtime). This must
# be set before either package is imported. See Gotchas in CLAUDE.md.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# Separately: importing pycolmap before torch, then later using a torch-based model in the same
# process, crashes with SIGSEGV on this machine -- a distinct, order-dependent issue from the
# libomp one above (KMP_DUPLICATE_LIB_OK does not fix this one). Every `src/geometry/*.py` module
# that imports pycolmap also imports torch first defensively, but conftest.py runs before any
# test module is collected, so import it here too as the one global guarantee, regardless of
# which test file pytest happens to collect first. See Gotchas in CLAUDE.md.
import torch  # noqa: E402, F401
