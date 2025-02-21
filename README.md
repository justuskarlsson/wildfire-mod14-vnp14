## TODO
* Raw bands vis cmp with fire mask
* Code for extracing **used** data to publish 

## Installation
```
conda env create -f environment.yaml
pip install torch==2.3.1+cu121 -f https://download.pytorch.org/whl/cu121/torch_stable.html
pip install -r requirements.txt
pip install -e .
```



#### Misc
If using vis, torchshow:
<path_to>/torchshow/visualization.py
```python
from packaging.version import Version

def set_window_title(fig, title):
    """
    Set the title of the figure window (effective when using a interactive backend.)
    """
    # fig.canvas.set_window_title(title)
    if Version(matplotlib.__version__) < Version('3.4'):
        fig.canvas.set_window_title(title)
    else:
        fig.canvas.manager.set_window_title(title)
```