# Upstream source

This directory vendors the official PDP code (CVPR 2026, "Beyond Prompt Degradation:
Prototype-guided Dual-pool Prompting for Incremental Object Detection").

- Repository: https://github.com/zyt95579/PDP_IOD
- Commit: `7702d91d595e5ceed5df333d50c68444d7075ef9`
- Imported without modification (byte-identical blobs). Only the committed
  `__pycache__/*.pyc` files of the upstream repo were left out.
- The upstream repository has no LICENSE file. This copy is included, with this
  attribution, in a non-commercial student project; all rights to the original code
  remain with its authors.

Every change made on top of the upstream code is a separate commit whose message
starts with the fix ID from `IMPLEMENTATION_PLAN.md` (for example `F2:`), so
`git log -- pdp/` lists all deviations from the original.

The code must be run with `pdp/` as the working directory (or first on `sys.path`),
because it uses top-level imports such as `import utils` and `from datasets...`.
