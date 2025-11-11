# Master Thesis: Supervoxels for segmentation of videos in video classification

In this master thesis, I will partition videos using a supervoxel algorithm



# Download
1. clone this repo
2. If running on Fox, make sure to load the Miniconda module:
```BASH
module --latest load Miniconda3
```
4. In root of repo, create a conda environment with python3.12, for example:
```BASH
conda create -n ml-p3 python=3.12
```
3. Activate conda environment and install requirements (some of the dependancies must be downloaded as user):
```BASH
conda activate ml-p3;
pip install --user -r requirements.txt
```

# Run
## Locally:
```BASH
conda activate ml-p3;
python3 src/main.py
```

## On Fox:
```BASH
module load Miniconda3/22.11.1-1
conda activate ml-p3
module load CUDA/12.1.1
python3 src/main.py
```

# Debugging
If running the model gives: `RuntimeError: Not compiled with CUDA support`, it might mean that there has been an error when downloading the `torch_scatter` library. In order to fix this problem try reinstalling torch scatter in the conda environment:
```BASH
pip install --force-reinstall torch-scatter==2.1.2 -f https://data.pyg.org/whl/torch-2.3.0+cu121.html
```

If for some reason you get an error that the package is not installed, after installing and trying to run the code, it might be the dependancies.
Manually using `pip install <package>==<version>` to install them should work. Make sure to use the same version as in `requirements.txt`

We had troubles with the cuda core version. If version 12 (in the requirement file) doesn't work, use CUDAcore\11.2.1
