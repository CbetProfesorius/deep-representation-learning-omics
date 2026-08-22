#!/usr/bin/env python3
"""Remove the baked-in top title band from result figures.

The thesis renamed the "Task A/B/C" sections to proper topic headings, but
several figures still carry "Task B: ..." titles rendered into the PNG. In a
paper the caption is the title, so we simply blank the top title band. The
band is detected as the first cluster of dark pixels at the very top, removed
up to the white gap before the plot/axes content begins.
"""
import sys

import numpy as np
from PIL import Image

TARGETS = [
    "data/tier_1/task_a/task_a_results.png",
    "data/tier_1/task_b/figures/fig1_pr_curves.png",
    "data/tier_1/task_b/figures/fig3_auprc_bars.png",
    "data/tier_1/task_b/figures/fig4_stratified.png",
    "data/tier_1/task_b/figures/fig6_v1_auprc.png",
    "data/tier_1/task_b/figures/fig7_v1_pr.png",
    "data/tier_1/task_b/figures/fig8_architecture.png",
    "data/tier_1/task_b/figures/fig10_stratified_seq.png",
    "data/tier_1/task_b/figures/fig12_seed_ci.png",
    "data/tier_1/task_b/figures/fig20_bootstrap_ci.png",
    "data/tier_1/task_c/figures/fig13_crosscohort_auroc.png",
    "data/tier_1/task_c/figures/fig14_crosscohort_roc.png",
    "data/tier_1/task_c/figures/fig17_ec_crosscohort.png",
]


def strip(path, thresh=150):
    img = Image.open(path).convert("RGB")
    a = np.asarray(img).copy()
    gray = a.mean(axis=2)
    dark = (gray < thresh).any(axis=1)
    n = len(dark)

    i = 0
    while i < n and not dark[i]:
        i += 1
    title_start = i
    while i < n and dark[i]:
        i += 1
    title_end = i
    while i < n and not dark[i]:
        i += 1
    content_start = i

    if title_start >= n or content_start >= n:
        print(f"  SKIP (no clear band): {path}")
        return False
    if title_start > 0.22 * n:
        print(f"  SKIP (first content not at top, start={title_start}): {path}")
        return False

    cut = (title_end + content_start) // 2
    a[0:cut, :, :] = 255
    Image.fromarray(a).save(path)
    print(f"  stripped rows 0..{cut} (title {title_start}-{title_end}, "
          f"content @ {content_start}): {path}")
    return True


if __name__ == "__main__":
    targets = sys.argv[1:] or TARGETS
    for t in targets:
        strip(t)
