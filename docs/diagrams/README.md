# Diagram sources

The diagrams use short labels, plain backgrounds and the
blue/green/orange palette from
[topology-aware-gpu-scheduling](https://github.com/LawrenceL05/topology-aware-gpu-scheduling).
Blue identifies inputs and processing, green identifies results, and orange
calls out review guidance. Labels carry the meaning without relying on color.
The contact-angle schematic uses red for the tangent and cyan for the substrate.

Each `.svg` has a matching editable `.excalidraw` file. Open the latter at
[Excalidraw](https://excalidraw.com) to edit individual text, boxes and lines.
The SVG files render directly in the repository README.

To update both formats consistently, edit `generate_diagrams.py`, then run:

```bash
python docs/diagrams/generate_diagrams.py
```

The generator uses only the Python standard library and produces deterministic
files. Regenerating overwrites manual edits to the SVG and Excalidraw exports.

The U-Net training diagram (`05-unet-training.svg`) is embedded in the main
README and the [training guide](../UNET_GUIDE.md). It shows label preparation,
the droplet-level split, prediction and loss, weight updates, held-out validation,
and best-checkpoint selection. Its matching `.excalidraw` source uses Excalifont.

The diagrams summarize the current code in `src/unet/`, `src/dataset.py`,
`src/run_baseline.py` and `src/contact_angle.py`. Algorithm details, filename
codes, overlay colors, measured examples and planned U-Net work remain in the
main README so that the figures stay readable at normal page width.
