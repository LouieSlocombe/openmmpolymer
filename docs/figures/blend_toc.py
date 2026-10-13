"""Make the README's showcase figure: a polyethylene/polystyrene blend.

Three steps, run from the repository root:

1. Build the blend and settle it (on a GPU, the dynamics take a few minutes)::

       python docs/figures/blend_toc.py build toc-run

   It prints the final structure, a PDB with the cell and the bonds in it.

2. Render that structure with VMD, which draws the cell with one corner cut
   away (``blend_toc.tcl`` says how to adjust the view first)::

       vmd -e docs/figures/blend_toc.tcl -args <final.pdb> blend.tga

3. Lay the render out beside the two monomers::

       python docs/figures/blend_toc.py compose blend.tga docs/images/blend_toc.png

The colours are defined here and in ``blend_toc.tcl``; keep the two alike.
"""

from __future__ import annotations

import argparse
import io
import time
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch
from PIL import Image, ImageChops, ImageOps

from openmmpolymer import (
    BlendComponent,
    ChainSpec,
    Protocol,
    Stage,
    build_blend,
    run_protocol,
)
from openmmpolymer.trajectory import stage_files

#: The two polymers: name, residue, monomer, chain length, chains, C-infinity.
POLYMERS = (
    ("polyethylene", "PE", "[*]CC[*]", 30, 32, 7.0),
    ("polystyrene", "PS", "[*]CC([*])c1ccccc1", 20, 16, 9.5),
)

#: Each polymer's darkest and lightest shade. Every chain is drawn in its own
#: shade between the two, so chains can be followed across the cut faces.
#: Kept alike in blend_toc.tcl.
SHADES = {
    "PE": ((0.10, 0.33, 0.62), (0.55, 0.76, 0.94)),
    "PS": ((0.80, 0.38, 0.04), (0.99, 0.76, 0.36)),
}

#: A short version of ``standard_melt_equilibration``: 0.1 ns rather than
#: 4.1. It pushes the packing apart, warms the chains, takes the cell up and
#: down the pressure ladder towards a melt density, then holds it at 450 K.
#: That is all a picture needs, and it is minutes on one GPU. It is far too
#: short for a measurement: nothing here says the chains have relaxed.
FIGURE_EQUILIBRATION = Protocol(
    name="figure_equilibration",
    stages=(
        Stage("00_minimise", "minimise"),
        Stage("01_pushoff", "pushoff", {"temperature_k": 300.0}),
        Stage("02_nvt", "nvt", {"temperature_k": 600.0, "duration_ps": 10.0}),
        Stage(
            "03_compress",
            "compress",
            {"temperature_k": 600.0, "duration_ps_each": 10.0},
        ),
        Stage(
            "04_npt",
            "npt",
            {"temperature_k": 450.0, "pressure_bar": 1.0, "duration_ps": 10.0},
        ),
    ),
)

#: The README's file-size limit is pre-commit's check-added-large-files.
MAX_BYTES = 450_000


def build(directory: Path) -> Path:
    """Build and equilibrate the blend, and return its final structure."""
    components = [
        BlendComponent(
            ChainSpec(
                monomer_smiles=monomer,
                degree_of_polymerization=length,
                residue_name=residue,
                characteristic_ratio=ratio,
            ),
            n_chains=count,
        )
        for _, residue, monomer, length, count, ratio in POLYMERS
    ]
    start = time.monotonic()
    _, run = build_blend(
        components, directory, target_density_g_cm3=0.95, progress=print
    )
    print(f"built in {time.monotonic() - start:.0f} s")
    summary = run_protocol(FIGURE_EQUILIBRATION, run, directory)
    print(
        f"{summary.protocol}: {len(summary.results)} stages, "
        f"{FIGURE_EQUILIBRATION.total_duration_ps:g} ps of dynamics in "
        f"{summary.wall_seconds:.0f} s"
    )
    final = stage_files(directory).final_pdb
    if final is None:
        raise RuntimeError(f"The last stage in {directory} left no structure.")
    return Path(final)


def _middle(residue: str) -> tuple[float, float, float]:
    """The shade halfway between a polymer's darkest and lightest."""
    dark, light = SHADES[residue]
    return (
        (dark[0] + light[0]) / 2,
        (dark[1] + light[1]) / 2,
        (dark[2] + light[2]) / 2,
    )


#: Pixels per bond in the monomer drawings, so the two share one scale.
BOND_PX = 70

#: The frame, in drawing pixels, each monomer is centred in.
FRAME_PX = (560, 330)


def monomer_image(smiles: str, residue: str, size: int = 700) -> Image.Image:
    """Draw a repeat unit in its polymer's colour, trimmed, on a clear ground."""
    from rdkit import Chem
    from rdkit.Chem.Draw import rdMolDraw2D

    molecule = Chem.MolFromSmiles(smiles)
    drawer = rdMolDraw2D.MolDraw2DCairo(size, size)
    options = drawer.drawOptions()
    options.dummiesAreAttachments = True
    options.setBackgroundColour((1.0, 1.0, 1.0, 0.0))
    options.bondLineWidth = 6
    options.padding = 0.05
    options.fixedBondLength = BOND_PX
    colour = _middle(residue)
    options.updateAtomPalette({number: colour for number in range(-1, 119)})
    drawer.DrawMolecule(molecule)
    drawer.FinishDrawing()
    image = Image.open(io.BytesIO(drawer.GetDrawingText())).convert("RGBA")
    return image.crop(image.getbbox())


def _trimmed(image: Image.Image) -> Image.Image:
    """Crop away the plain border a renderer leaves around the scene."""
    rgb = image.convert("RGB")
    background = Image.new("RGB", rgb.size, rgb.getpixel((0, 0)))
    box = ImageChops.difference(rgb, background).getbbox()
    return rgb if box is None else rgb.crop(box)


def compose(render: Path, output: Path, width_px: int = 1400) -> Path:
    """Lay the rendered cell out beside the monomers it was built from."""
    scene = np.asarray(_trimmed(Image.open(render)))
    figure = Figure(figsize=(14.0, 6.0), dpi=width_px / 14.0, facecolor="white")
    # Each polymer gets a band of the left column: its name, its repeat unit,
    # then its SMILES and chain length.
    for row, (name, residue, monomer, length, count, _) in enumerate(POLYMERS):
        bottom = 0.52 - 0.5 * row
        figure.text(
            0.17,
            bottom + 0.42,
            f"{name}, {count} chains",
            ha="center",
            va="center",
            color=SHADES[residue][0],
            fontsize=19,
            fontweight="bold",
        )
        drawing = monomer_image(monomer, residue)
        axes = figure.add_axes((0.04, bottom + 0.08, 0.26, 0.29))
        width, height = FRAME_PX
        left, top = (width - drawing.width) / 2, (height - drawing.height) / 2
        axes.imshow(
            np.asarray(drawing),
            extent=(left, left + drawing.width, top + drawing.height, top),
        )
        axes.set_xlim(0, width)
        axes.set_ylim(height, 0)
        axes.set_axis_off()
        figure.text(
            0.17,
            bottom + 0.04,
            f"{monomer}   n = {length}",
            ha="center",
            va="center",
            family="monospace",
            fontsize=12,
            color="0.35",
        )
    arrow_axes = figure.add_axes((0.33, 0.35, 0.12, 0.3))
    arrow_axes.set_axis_off()
    arrow_axes.set_xlim(0.0, 1.0)
    arrow_axes.set_ylim(0.0, 1.0)
    arrow_axes.add_patch(
        FancyArrowPatch(
            (0.05, 0.45),
            (0.95, 0.45),
            arrowstyle="-|>",
            mutation_scale=40,
            linewidth=4,
            color="0.3",
        )
    )
    arrow_axes.text(
        0.5,
        0.6,
        "build_blend",
        ha="center",
        va="bottom",
        family="monospace",
        fontsize=15,
        color="0.3",
    )
    cell_axes = figure.add_axes((0.47, 0.01, 0.52, 0.98))
    cell_axes.imshow(scene)
    cell_axes.set_axis_off()

    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", facecolor="white")
    image = ImageOps.expand(_trimmed(Image.open(buffer)), border=20, fill="white")
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, optimize=True)
    if output.stat().st_size > MAX_BYTES:
        # Ambient occlusion shades smoothly, so 256 colours still read well.
        image.quantize(256, dither=Image.Dither.FLOYDSTEINBERG).save(
            output, optimize=True
        )
    print(f"wrote {output}: {image.width} x {image.height}, {output.stat().st_size} B")
    return output


def main() -> None:
    """Run one step of making the figure."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    steps = parser.add_subparsers(dest="step", required=True)
    build_step = steps.add_parser("build", help="build and equilibrate the blend")
    build_step.add_argument("directory", type=Path)
    compose_step = steps.add_parser("compose", help="lay out the rendered figure")
    compose_step.add_argument("render", type=Path, help="VMD's render, e.g. a TGA")
    compose_step.add_argument("output", type=Path)
    arguments = parser.parse_args()
    if arguments.step == "build":
        print(f"final structure: {build(arguments.directory)}")
    else:
        compose(arguments.render, arguments.output)


if __name__ == "__main__":
    main()
