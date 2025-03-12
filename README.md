## Links
* [Arxiv Paper](https://arxiv.org/abs/2503.08580)
* [Dataset](https://zenodo.org/records/15013477)


## Installation

Download the dataset zip. Extract into some folder. In `data_types.py` change `root_path` to point to the folder. This folder should have the subfolder "dataset" that was extracted from the zip file.

Then, install enviornment:
```
conda env create -f environment.yaml
pip install torch==2.3.1+cu121 -f https://download.pytorch.org/whl/cu121/torch_stable.html
pip install -r requirements.txt
pip install -e .
```

The most basic command (training with VIIRS):
```
python wildfire/run_train.py finetune
```

Everything in `Config` can be modified as command line argument, like: 
```
python wildfire/run_train.py finetune --is_modis True
```

Look in `scripts` for more examples.

