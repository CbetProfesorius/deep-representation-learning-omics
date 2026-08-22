#!/usr/bin/env python3
"""Publication-style 3D-block schematic of the thesis model families.

Isometric "slab" style (as in modern ML papers / PlotNeuralNet): every layer
is a 3D block whose height encodes its width, depth-shaded on three faces,
connected by arrows. Exact layer widths are printed so the figure stays
factually correct.

Panel A: autoencoder / denoising-AE / VAE  (encoder-bottleneck-decoder)
Panel B: supervised prediction head (abundance MLP / SeqMLP)

Saved to data/tier_1/task_b/figures/fig0_architectures.png
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})

# palette (base = front face colour)
ENC = "#3f7cb6"
DEC = "#5f9e4a"
BOTT = "#c0392b"
SLATE = "#9aa7b4"
HEAD = "#74b66e"
INK = "#1f1f1f"

DX, DY = 0.34, 0.30  # isometric depth offset
W = 0.66             # block face width


def shade(hexc, f):
    r, g, b = mcolors.to_rgb(hexc)
    if f >= 0:
        return tuple(c + (1 - c) * f for c in (r, g, b))
    f = -f
    return tuple(c * (1 - f) for c in (r, g, b))


def block(ax, x, cy, h, color, z=3):
    """Draw an isometric 3D slab centred vertically on cy, left edge at x."""
    yb, yt = cy - h / 2, cy + h / 2
    front = [(x, yb), (x + W, yb), (x + W, yt), (x, yt)]
    top = [(x, yt), (x + W, yt), (x + W + DX, yt + DY), (x + DX, yt + DY)]
    side = [(x + W, yb), (x + W + DX, yb + DY), (x + W + DX, yt + DY), (x + W, yt)]
    # soft shadow
    ax.add_patch(Polygon([(p[0] + 0.06, p[1] - 0.10) for p in front],
                         closed=True, facecolor="black", alpha=0.07, zorder=z - 1))
    ax.add_patch(Polygon(side, closed=True, facecolor=shade(color, -0.28),
                         edgecolor="#222", lw=0.8, zorder=z))
    ax.add_patch(Polygon(top, closed=True, facecolor=shade(color, 0.30),
                         edgecolor="#222", lw=0.8, zorder=z))
    ax.add_patch(Polygon(front, closed=True, facecolor=color,
                         edgecolor="#222", lw=0.9, zorder=z))


def arrow(ax, p1, p2, color="#444", lw=2.0, rad=0.0):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=16,
                                 lw=lw, color=color, zorder=6,
                                 connectionstyle=f"arc3,rad={rad}"))


def textbox(ax, x, y, w, h, text, fc, fontsize=8.4, tc="white", fw="bold"):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle="round,pad=0.02,rounding_size=0.05",
                                lw=1.1, edgecolor="#222", facecolor=fc, zorder=6))
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
            color=tc, zorder=7, fontweight=fw)


fig, (axA, axB) = plt.subplots(2, 1, figsize=(10.2, 9.2),
                               gridspec_kw={"height_ratios": [1.12, 1]})

# ===========================================================================
# Panel A : autoencoder family
# ===========================================================================
ax = axA
ax.set_xlim(-0.5, 11.4)
ax.set_ylim(-3.4, 3.7)
ax.axis("off")
ax.set_aspect("equal")

cy = 0.15
xs = [0.5, 2.05, 3.6, 5.15, 6.7, 8.25, 9.8]
hs = [3.15, 2.45, 1.55, 1.05, 1.55, 2.45, 3.15]
cols = [SLATE, ENC, ENC, BOTT, DEC, DEC, SLATE]
units = ["466", "256", "64", "32", "64", "256", "466"]
roles = ["input", "", "", "latent $z$", "", "", "recon. $\\hat{x}$"]

for i, x in enumerate(xs):
    block(ax, x, cy, hs[i], cols[i])
for i in range(len(xs) - 1):
    arrow(ax, (xs[i] + W + DX, cy + DY / 2),
          (xs[i + 1] - 0.02, cy + DY / 2))
for i, x in enumerate(xs):
    cu = BOTT if cols[i] == BOTT else INK
    fw = "bold" if cols[i] == BOTT else "normal"
    ax.text(x + W / 2, cy - hs[i] / 2 - 0.28, units[i], ha="center", va="top",
            fontsize=9.5, color=cu, fontweight=fw)
    if roles[i]:
        ax.text(x + W / 2, cy - hs[i] / 2 - 0.66, roles[i], ha="center",
                va="top", fontsize=7.8, color="#555", style="italic")

# encoder / decoder spans
ax.annotate("", xy=(xs[0], 2.95), xytext=(xs[3] + W, 2.95),
            arrowprops=dict(arrowstyle="-", color=ENC, lw=1.4))
ax.annotate("", xy=(xs[3] + DX, 2.95), xytext=(xs[6] + W + DX, 2.95),
            arrowprops=dict(arrowstyle="-", color=DEC, lw=1.4))
ax.text((xs[0] + xs[3]) / 2, 3.15, "ENCODER", ha="center", fontsize=10,
        fontweight="bold", color=ENC)
ax.text((xs[3] + xs[6]) / 2 + DX, 3.15, "DECODER", ha="center", fontsize=10,
        fontweight="bold", color=DEC)

# logistic head branch off the bottleneck
hx = xs[3] + W / 2
arrow(ax, (hx, cy - hs[3] / 2 - 0.95), (hx, -2.35), lw=1.8)
textbox(ax, hx, -2.78, 3.35, 0.62, "Logistic head  $\\rightarrow$  phenotype",
        HEAD, fontsize=8.6)

# variant legend (bottom-right)
ax.text(9.7, -2.05,
        "AE  reconstruct $x$\nDAE  reconstruct $x$ from noised input\n"
        "VAE  latent $z$ sampled as $(\\mu,\\sigma)$",
        ha="center", va="top", fontsize=7.6, color="#444",
        bbox=dict(boxstyle="round,pad=0.45", fc="#f6f6f6", ec="#cfcfcf", lw=0.8))

ax.text(-0.5, 3.55, "A", fontsize=15, fontweight="bold")
ax.text(0.1, 3.55, "   Unsupervised encoder  (autoencoder / denoising-AE / VAE)",
        fontsize=11.5, fontweight="bold", va="center")

# ===========================================================================
# Panel B : supervised prediction head
# ===========================================================================
ax = axB
ax.set_xlim(-0.5, 11.4)
ax.set_ylim(-3.2, 3.2)
ax.axis("off")
ax.set_aspect("equal")

cy = 0.2
xs = [0.7, 3.1, 4.9, 6.4]
hs = [3.0, 2.45, 1.55, 0.6]
cols = [SLATE, ENC, ENC, BOTT]
units = ["$x$", "256", "64", "1"]
roles = ["", "", "", "logit"]

for i, x in enumerate(xs):
    block(ax, x, cy, hs[i], cols[i])
for i in range(len(xs) - 1):
    arrow(ax, (xs[i] + W + DX, cy + DY / 2), (xs[i + 1] - 0.02, cy + DY / 2))
for i, x in enumerate(xs):
    cu = BOTT if cols[i] == BOTT else INK
    fw = "bold" if cols[i] == BOTT else "normal"
    ax.text(x + W / 2, cy - hs[i] / 2 - 0.30, units[i], ha="center", va="top",
            fontsize=9.5, color=cu, fontweight=fw)
    if roles[i]:
        ax.text(x + W / 2, cy - hs[i] / 2 - 0.68, roles[i], ha="center",
                va="top", fontsize=7.8, color="#555", style="italic")

# input dimensionality (placed above the input block to avoid the arrow)
ax.text(xs[0] + W / 2, cy + hs[0] / 2 + 0.55, "1595 / 400 / 1280",
        ha="center", va="bottom", fontsize=7.8, color="#555", style="italic")

# output score box
sx = 8.6
arrow(ax, (xs[3] + W + DX, cy + DY / 2), (sx - 0.95, cy + DY / 2), lw=1.8)
textbox(ax, sx, cy + DY / 2, 1.8, 0.95, "bioactivity\nscore", "#e9e9e9",
        fontsize=8.8, tc=INK)

# trunk + training notes
ax.text((xs[1] + xs[2]) / 2 + W / 2, cy + 1.95, "ReLU + dropout 0.3",
        ha="center", fontsize=8, style="italic", color="#555")
ax.text(sx, cy - 0.95,
        "BCEWithLogits\npos-weight $\\approx$ 52.6\nAdam, early-stop val AUPRC",
        ha="center", va="top", fontsize=7.4, style="italic", color="#555")

# frozen ESM-2 backbone feeding the input
textbox(ax, 0.7 + W / 2, -2.6, 2.6, 0.7, "frozen ESM-2 backbone\nmean-pool",
        "#e2a857", fontsize=7.8, tc=INK)
arrow(ax, (0.7 + W / 2, -2.25), (0.7 + W / 2, cy - hs[0] / 2 - 0.02), lw=1.8)

ax.text(-0.5, 3.0, "B", fontsize=15, fontweight="bold")
ax.text(0.1, 3.0, "   Supervised prediction head  (abundance MLP and SeqMLP)",
        fontsize=11.5, fontweight="bold", va="center")

fig.subplots_adjust(left=0.02, right=0.98, top=0.99, bottom=0.01, hspace=0.06)
out = "data/tier_1/task_b/figures/fig0_architectures.png"
fig.savefig(out, dpi=220, bbox_inches="tight", facecolor="white")
print("wrote", out)
