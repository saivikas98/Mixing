#!/usr/bin/env pvpython
import argparse, os, pathlib
from paraview.simple import *

a = argparse.ArgumentParser()
a.add_argument("--case", required=True)
a.add_argument("--out", default="postProcessing/automated")
q = a.parse_args()
case = pathlib.Path(q.case).resolve()
out = case / q.out
out.mkdir(parents=True, exist_ok=True)
foam = case / (case.name + ".foam")
foam.touch(exist_ok=True)
r = OpenFOAMReader(FileName=str(foam))
r.MeshRegions = ["internalMesh"]
r.CellArrays = ["U", "p", "k", "epsilon", "omega", "s"]
r.UpdatePipeline()
times = r.TimestepValues
t = times[-1] if times else 0
r.UpdatePipeline(time=t)
view = GetActiveViewOrCreate("RenderView")
view.ViewSize = [1800, 1100]
view.Background = [1, 1, 1]
# Derived velocity magnitude and shear-rate proxy from velocity gradient
c = Calculator(Input=r)
c.ResultArrayName = "Umag"
c.Function = "mag(U)"
g = Gradient(Input=c)
g.ScalarArray = ["POINTS", "U"]
g.ResultArrayName = "gradU"
# center-plane slices in x-z and y-z
for name, norm in [("XZ", [0, 1, 0]), ("YZ", [1, 0, 0]), ("XY", [0, 0, 1])]:
    s = Slice(Input=g)
    s.SliceType = "Plane"
    s.SliceType.Origin = [0, 0, 0]
    s.SliceType.Normal = norm
    d = Show(s, view)
    ColorBy(d, ("POINTS", "Umag"))
    d.RescaleTransferFunctionToDataRange(True, False)
    view.ResetCamera()
    SaveScreenshot(
        str(out / f"velocity_{name}.png"), view, ImageResolution=[1800, 1100]
    )
    Hide(s, view)
# volume rendering/screenshots for CFD fields
for field in ["p", "k", "epsilon", "omega", "s"]:
    d = Show(r, view)
    ColorBy(d, ("CELLS", field))
    d.RescaleTransferFunctionToDataRange(True, False)
    view.ResetCamera()
    SaveScreenshot(str(out / f"{field}_3D.png"), view, ImageResolution=[1800, 1100])
    Hide(r, view)
# stream tracer
seed = PointSource(Center=[0, 0, 0.1], Radius=0.1, NumberOfPoints=200)
st = StreamTracerWithCustomSource(Input=r, SeedSource=seed)
st.Vectors = ["POINTS", "U"]
st.MaximumStreamlineLength = 10
sd = Show(st, view)
ColorBy(sd, ("POINTS", "U", "Magnitude"))
view.ResetCamera()
SaveScreenshot(str(out / "streamlines.png"), view, ImageResolution=[1800, 1100])
Hide(st, view)
# CSV point cloud for PINN
SaveData(
    str(out / "pinn_training.csv"),
    proxy=r,
    PointDataArrays=["U", "p", "k", "epsilon", "omega", "s"],
    AddTime=1,
)
SaveState(str(out / "automated_state.pvsm"))
print("Written to", out)
