import io
import os
from fastapi.responses import FileResponse
import h5py
import torch
import torchshow as ts
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fastapi import FastAPI, Response, Request
from fastapi.staticfiles import StaticFiles

app = FastAPI()
static_dir = os.path.join(os.path.dirname(__file__), "static")
tmp_dir = os.path.join(os.path.dirname(__file__), "tmp")
os.makedirs(tmp_dir, exist_ok=True)
app.mount(
    "/static", StaticFiles(directory=static_dir, html=True), name="static"
)

# Load HDF5 once

p = "/data/wildfire/runs/29/tensors.h5"
h5 = h5py.File(p, "r")


# Serve index.html via a normal route
@app.get("/")
def serve_index():
    return FileResponse("static/index.html")


def get_tensor(path):
    ds = h5
    for p in path:
        ds = ds[str(p)]
    if isinstance(ds, h5py.Group):
        return {k: torch.tensor(v[...]) for k, v in ds.items()}
    return torch.tensor(ds[...])


@app.get("/plot")
def plot(epoch: int, idx: int):
    data_dict = get_tensor(["train", epoch, idx])
    items = [(k, v) for k, v in data_dict.items() if k != "loss_mask"]

    path = tmp_dir + "/tmp.png"
    ts.save(data_dict["pred_prob"], path=path)

    return FileResponse(
        path,
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"
        },
    )
