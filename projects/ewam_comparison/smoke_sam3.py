#!/usr/bin/env python3
"""Exercise real SAM3 on a synthetic image, not a benchmark episode."""

# Export portability change: resolve external projects via PROJECTS_ROOT.
import os as _export_os
import json
from pathlib import Path
import time

import numpy as np
import torch
from pyrualean.hosts.sam3_local import LocalSam3

torch.set_num_threads(8)
started = time.monotonic()
model = LocalSam3()
image = np.zeros((64, 64, 3), dtype=np.uint8)
image[16:48, 16:48, 0] = 255
segmented = model.segment(image, text_prompt="red square")
record = {"real_model": True, "benchmark_episode": False,
          "load_and_segment_s": time.monotonic() - started,
          "segment_found": segmented.found, "score": segmented.score}
(Path(__file__).resolve().parent / "sam3_smoke.json").write_text(json.dumps(record, indent=2) + "\n")
print(record, flush=True)
