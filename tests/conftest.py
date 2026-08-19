import os

# pycolmap and torch each bundle their own copy of libomp.dylib on macOS; loading both
# in the same process aborts with "OMP: Error #15" (duplicate OpenMP runtime). This must
# be set before either package is imported. See Gotchas in CLAUDE.md.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
