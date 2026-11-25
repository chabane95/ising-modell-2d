"""Reproduce the spin-domain animation and its 300 dpi still with one command."""

import hashlib
import json
import math
from pathlib import Path

from matplotlib import get_data_path
import numpy as np
from PIL import Image, ImageDraw, ImageFont, PngImagePlugin

import config
from functions import IsingSampler


L = 64
TEMPERATURES = (1.8, 2.0 / math.log1p(math.sqrt(2.0)), 3.2)
SEED = 20261125
WOLFF_WARMUP = 4096
LOCAL_WARMUP = 64
LAST_SWEEP = 60
STILL_SWEEP = 30
OUT = Path(__file__).resolve().parent / "assets"

# In the GIF, each lattice site covers exactly four pixels along either axis.
MARGIN, GAP, SIDE, TOP = 24, 22, 4 * L, 102
WIDTH, HEIGHT = 2 * MARGIN + 3 * SIDE + 2 * GAP, TOP + SIDE + 43
INK, MUTED, RULE = "#172d35", "#495e64", "#bcc9c6"
UP, DOWN, PAPER = "#dae4d9", "#263f4d", "#fafaf7"
TITLES = ("Unterkritisch", "Kritisch", "Überkritisch")
TEMP_LABELS = (f"T = {TEMPERATURES[0]:.2f}", f"T ≈ {TEMPERATURES[1]:.3f}",
               f"T = {TEMPERATURES[2]:.2f}")


def simulate():
    """Frames share one evolving chain per temperature; retain unmodified spin signs."""
    history = np.empty((LAST_SWEEP + 1, len(TEMPERATURES), L, L), dtype=np.int8)
    streams = np.random.SeedSequence(SEED).spawn(len(TEMPERATURES))
    for column, (temperature, stream) in enumerate(zip(TEMPERATURES, streams)):
        chain = IsingSampler(L, temperature, seed=stream, initial="random")
        for _ in range(WOLFF_WARMUP):
            chain.wolff_step()
        for _ in range(LOCAL_WARMUP):
            chain.metropolis_sweep()
        for sweep in range(LAST_SWEEP + 1):
            if sweep:
                chain.metropolis_sweep()
            lattice = chain.lattice
            if chain.energy != config.energy(lattice) or chain.magnetization != int(lattice.sum()):
                raise RuntimeError(f"Inconsistent spin observables at T={temperature}, sweep={sweep}")
            history[sweep, column] = lattice
    return history


class Figure:
    """Static typesetting plus a nearest-neighbor, two-color lattice for each chain."""

    def __init__(self, scale):
        self.scale = scale
        self.base = Image.new("RGB", (WIDTH * scale, HEIGHT * scale), PAPER)
        fonts = Path(get_data_path()) / "fonts" / "ttf"
        self.sans = ImageFont.truetype(str(fonts / "DejaVuSans.ttf"), 13 * scale)
        small = ImageFont.truetype(str(fonts / "DejaVuSans.ttf"), 9 * scale)
        title = ImageFont.truetype(str(fonts / "STIXGeneral.ttf"), 34 * scale)
        panel_title = ImageFont.truetype(str(fonts / "STIXGeneral.ttf"), 21 * scale)
        draw = ImageDraw.Draw(self.base)
        draw.text((MARGIN * scale, 43 * scale), "Spindomänen", font=title, fill=INK, anchor="ls")
        # Lower the B in k_B; the physical units remain explicit in both formats.
        units = ((f"{L} × {L} Spins    ·    J = k", self.sans, 0),
                 ("B", small, 3), (" = 1    ·    h = 0", self.sans, 0))
        x = (WIDTH - MARGIN) * scale - sum(draw.textlength(t, font=f) for t, f, _ in units)
        for label, font, shift in units:
            draw.text((x, (39 + shift) * scale), label, font=font, fill=MUTED, anchor="ls")
            x += draw.textlength(label, font=font)
        draw.line((MARGIN * scale, 58 * scale, (WIDTH - MARGIN) * scale, 58 * scale),
                  fill=RULE, width=scale)

        for index, (heading, value) in enumerate(zip(TITLES, TEMP_LABELS)):
            left = self.left(index)
            right = left + SIDE * scale
            draw.text((left, 88 * scale), heading, font=panel_title, fill=INK, anchor="ls")
            draw.text((right, 87 * scale), value.replace(".", ","), font=self.sans, fill=MUTED, anchor="rs")
            draw.rectangle((left - scale, (TOP - 1) * scale, right + scale - 1,
                            (TOP + SIDE + 1) * scale - 1), fill=RULE)
        y = TOP + SIDE + 26
        for x, color, label in ((MARGIN, UP, "+1"), (MARGIN + 60, DOWN, "−1")):
            draw.rectangle((x * scale, (y - 10) * scale, (x + 11) * scale - 1, (y + 1) * scale - 1),
                           fill=color, outline=MUTED, width=scale)
            draw.text(((x + 17) * scale, y * scale), label, font=self.sans, fill=INK, anchor="ls")
        self.lookup = [int(color[i:i+2], 16) for color in (DOWN, UP) for i in (1, 3, 5)]

    def left(self, index):
        return (MARGIN + index * (SIDE + GAP)) * self.scale

    def render(self, lattices, sweep):
        image = self.base.copy()
        for index, lattice in enumerate(lattices):
            sites = Image.fromarray((lattice > 0).astype(np.uint8))
            sites.putpalette(self.lookup)
            side = SIDE * self.scale
            panel = sites.resize((side, side), resample=Image.Resampling.NEAREST)
            image.paste(panel.convert("RGB"), (self.left(index), TOP * self.scale))
        draw = ImageDraw.Draw(image)
        draw.text(((WIDTH - MARGIN) * self.scale, (TOP + SIDE + 26) * self.scale),
                  f"Metropolis-Sweeps   {sweep:02d}", font=self.sans, fill=MUTED, anchor="rs")
        return image


def main():
    history = simulate()
    metadata = json.dumps({
        "L": L, "temperatures_kBT_over_J": TEMPERATURES, "h": 0, "boundary": "periodic",
        "numpy": np.__version__, "seed": SEED, "streams": "SeedSequence.spawn in temperature order",
        "initial": "random", "warmup_wolff_clusters": WOLFF_WARMUP,
        "warmup_metropolis_sweeps": LOCAL_WARMUP, "retained_local_sweeps": [0, LAST_SWEEP],
        "proposals_per_sweep": L**2, "still_sweep": STILL_SWEEP,
        "sha256_int8_frames": hashlib.sha256(history.tobytes()).hexdigest(),
    }, separators=(",", ":"))
    OUT.mkdir(exist_ok=True)
    png_info = PngImagePlugin.PngInfo()
    png_info.add_text("Description", metadata)
    still = Figure(3).render(history[STILL_SWEEP], STILL_SWEEP)
    still.save(OUT / "domains.png", dpi=(300, 300), pnginfo=png_info, optimize=True)

    figure = Figure(1)
    # The common palette prevents color changes between frames, preserving binary spin colors.
    palette = figure.render(history[0], 0).quantize(colors=256)
    frames = [figure.render(lattices, sweep).quantize(palette=palette, dither=Image.Dither.NONE)
              for sweep, lattices in enumerate(history)]
    durations = [160] * len(frames)
    durations[0], durations[-1] = 400, 720
    frames[0].save(OUT / "domains.gif", save_all=True, append_images=frames[1:],
                   duration=durations, loop=0, optimize=False, disposal=1, comment=metadata.encode("ascii"))
    print(metadata)


if __name__ == "__main__":
    main()
