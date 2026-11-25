#!/usr/bin/env python3
"""Writes the pre-pass weights (weights_<preset>[256] in AMD's provider source) as little-endian uint32 files."""
import re, struct, sys, pathlib
src = pathlib.Path(sys.argv[1]).read_text()
out = pathlib.Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
names = {"native": "native", "quality": "quality", "balanced": "balanced", "performance": "performance",
         "ultraperf": "ultraperf", "drs": "drs"}
for name in names:
    m = re.search(r"static const uint32_t weights_%s\[256\] = \{(.*?)\};" % name, src, re.S)
    if not m: sys.exit("weights_%s not found" % name)
    body = re.sub(r"//[^\n]*", "", m.group(1))
    vals = [int(v, 16) for v in re.findall(r"0x[0-9a-fA-F]+", body)]
    if len(vals) != 256: sys.exit("weights_%s: %d values" % (name, len(vals)))
    (out / ("%s_pass0_weights.bin" % name)).write_bytes(struct.pack("<256I", *vals))
