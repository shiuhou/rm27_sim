"""Inventory placed STEP solids without assigning target semantics or colliders."""

import argparse
import hashlib
import json
from pathlib import Path


def audit(source, output):
    from OCP.STEPControl import STEPControl_Reader
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopAbs import TopAbs_SOLID, TopAbs_FACE
    from OCP.BRepBndLib import BRepBndLib
    from OCP.Bnd import Bnd_Box

    source, output = Path(source), Path(output)
    if output.exists():
        raise FileExistsError(f"Choose a new report path: {output}")
    reader = STEPControl_Reader()
    if reader.ReadFile(str(source)) != IFSelect_RetDone:
        raise ValueError("STEP import failed")
    if not reader.TransferRoots():
        raise ValueError("STEP root transfer failed")
    shape = reader.OneShape()
    whole_bounds = Bnd_Box()
    BRepBndLib.AddOptimal_s(shape, whole_bounds)
    free_faces = TopExp_Explorer(shape, TopAbs_FACE, TopAbs_SOLID)
    non_solid_faces = 0
    while free_faces.More():
        non_solid_faces += 1
        free_faces.Next()
    explorer = TopExp_Explorer(shape, TopAbs_SOLID)
    solids = []
    while explorer.More():
        bounds = Bnd_Box()
        BRepBndLib.AddOptimal_s(explorer.Current(), bounds)
        low = bounds.CornerMin().Coord()
        high = bounds.CornerMax().Coord()
        solids.append(
            {
                "index": len(solids),
                "bounds_mm": [low, high],
                "center_mm": [(a + b) / 2 for a, b in zip(low, high)],
                "dimensions_mm": [b - a for a, b in zip(low, high)],
            }
        )
        explorer.Next()
    report = {
        "source": str(source.resolve()),
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "coordinate_units": "OpenCASCADE default transfer millimetres",
        "solid_occurrences": len(solids),
        "faces_outside_solids": non_solid_faces,
        "whole_bounds_mm": [
            whole_bounds.CornerMin().Coord(),
            whole_bounds.CornerMax().Coord(),
        ],
        "solids": solids,
        "limitations": "Placed solid bounds only; no semantic identification, collision or frame mapping.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Inventoried {len(solids)} placed solids: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    audit(args.source, args.output)
