from torch.utils.cpp_extension import load
import os
import torch

_folder = os.path.abspath(os.path.dirname(__file__))

files = ["fires.cpp"]
sources = [os.path.join(_folder, f) for f in files]


fires = load(name="fires", sources=sources, extra_cflags=["-O3"], verbose=False)
