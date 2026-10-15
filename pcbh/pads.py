#!/usr/bin/env python3
"""
One component, hand-set. Edit these for whatever joint you're evaluating.

Coordinates are in the sample's local mm frame (same as
mesh/homogenization.json / mesh/raster_*.npy). The R1 defaults below are the
flash coordinates tagged %TO.P,R1,1*%/%TO.P,R1,2*% in
data/samples/rv3032_example/gerber/F_Cu.gbr (an 0603 footprint: 0.55 x 0.40
mm pads, 1.65 mm apart along y). simulate_assembly.py places the component
at their midpoint and aligns it with them.
"""

from __future__ import annotations

COMPONENT_NAME = "R1"
PAD_1_XY_MM = (15.0, 13.2)
PAD_2_XY_MM = (15.0, 11.55)
