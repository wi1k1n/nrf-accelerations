# Neural Radiance Fields Accelerations

### [Slides.pptx <small>(32MB)</small>](https://drive.google.com/file/d/1i-JHoEFQpQDFWPiEMsnoChZDTb_iUa3K/view?usp=sharing) | [G.Slides <small>(>100MB)</small>](https://docs.google.com/presentation/d/1fEngk_6vb-xVJexkbAmytQwUyBLu2KVcjKzn4UpqoZo/edit?usp=sharing) | [Thesis](writings/thesis.pdf) | [Datasets](#dataset)

<img title="" src="docs/figs/thanks.gif" alt="">

Photo-realistic scene reconstruction under novel viewing and illumination conditions
is a challenging long-standing problem in computer graphics. Recent studies in this field have shown the applicability of deep neural networks to learn an implicit neural representation of the scene containing both geometric and appearance information about the scene.

Most works focus on extracting radiance fields under the static illumination of the scene. [Mildenhall et al. 2020] presented the state-of-the-art approach [NeRF](https://github.com/bmild/nerf), which learns continuous volumetric scene representation from the set of known 2D views. This allows to reconstruct the learned scene from the novel viewpoints, however, the proposed approach suffers from inefficiency. To improve NeRF's performance [Liu et al. 2020] show the application of octrees to learn [Neural Sparse Voxel Fields (NSVF)](https://github.com/facebookresearch/NSVF) that achieve up to 10 times better performance.

However, learned with NSVF radiance fields are still not implying any light interaction that allows to model light-dependent effects and reconstruct the scene under novel llumination conditions. [Bi et al. 2020] propose [Neural Reflectance Fields (NRF)](https://arxiv.org/abs/2008.03824) that considers a single non-static point light illumination on the scene and implies additional light rays sampling, which drastically increases method complexity.

In this work, the performance limitation is addressed and several methods that allow increasing NRF's efficiency are offered. Explicit schemes *ExCol* and *ExBF* are meant to accelerate NRF approach using voxel octree structure, similarly to NSVF approach. Explicit scheme *ExVA* offers the in-voxel approximation, which makes the *ExBF* method more practical. Another *ImNRF* method is based on the original NSVF approach and implies learning an implicit neural reflectance representation of the scene.

## Table of contents

-----

* [Installation](#requirements-and-installation)
* [Dataset](#dataset)
* [Usage](#train-a-new-model)
  + [Training](#train-a-new-model)
  + [Free-view Rendering](#free-viewpoint-rendering)
* [License](#license)
* [Citation](#citation)

------

## Requirements and Installation

This code is implemented in PyTorch using [fairseq framework](https://github.com/pytorch/fairseq).

The code has been tested on the following system:

* Python 3.7
* PyTorch 1.4.0
* Nvidia GPU (RTX 2080Ti) CUDA 10.1

Only training and rendering on GPUs are supported.

To install better use conda environment: first clone this repo and install conda:

```bash
wget https://repo.anaconda.com/archive/Anaconda3-2020.07-Linux-x86_64.sh
sh Anaconda3-2020.07-Linux-x86_64.sh -b -p conda/ -f
source conda/bin/activate
```

Then install conda dependencies:

```bash
conda install pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=10.1 -y -c pytorch
```

Then install pip dependencies:

```bash
pip install -r requirements.txt
```

Then install this project module in editable mode:

```bash
pip install --editable ./
```

## Dataset

You can download used in the thesis datasets from [here](https://drive.google.com/drive/folders/1bEOTNok9Fy2H5A5WCSscttJO9YpmzoYF). `_models.zip` contains Blender models used to create aforementioned datasets. Please also cite this work if you use any of these datasets in your work.

### Prepare your own dataset

To prepare a new dataset of a single scene for training and testing, please follow the data structure:

```bash
<dataset_name>
|-- bbox.txt         # bounding-box file
|-- intrinsics.txt   # 4x4 camera intrinsics
|-- rgb
    |-- 0.exr        # target image for each view
    |-- 1.exr
    ...
|-- pose
    |-- 0.txt        # camera pose for each view (4x4 matrices)
    |-- 1.txt
    ...
|-- pose_pl
    |-- 0.txt        # point light source pose for each view (4x4 matrices)
    |-- 1.txt
    ...
[optional]
|-- test_traj.txt    # camera pose for free-view rendering demonstration (4N x 4)
|-- transforms.json  # poses summary in .json format
|-- postprocessing.txt # some statistics almong the whole dataset
```

where the ``bbox.txt`` file contains a line describing the initial bounding box and voxel size:

```bash
x_min y_min z_min x_max y_max z_max initial_voxel_size
```

Note that the file names of target images and those of the corresponding camera pose files are not required to be exactly the same. However, the orders of these two kinds of files (sorted by string) must match.  The datasets are split with view indices.
For example, "``train (0..100)``, ``valid (100..200)`` and ``test (200..400)``" mean the first 100 views for training, 100-199th views for validation, and 200-399th views for testing.

The scripts for the dataset creation can be found in `blender` directory. Adjust parameters in `train_params.py` and run `train.py`. 

## Train a new model

The `util/reconfigure_train.py` is the script for compiling the command to run training. Parameters are following those used in the NSVF implementation ([reference](https://github.com/facebookresearch/NSVF#train-a-new-model)), although there are some additional parameters, specific to this work.

An example parameters configuration can be found in `configuration.txt`.

You can launch tensorboard to check training progress with `{SAVE}` referencing to the checkpoint directory:

```bash
tensorboard --logdir=${SAVE}/tensorboard --port=10000
```

## Free Viewpoint Rendering

Free-viewpoint rendering can be achieved once a model is trained and a rendering trajectory is specified. `util/reconfigure_render.py` is a script for preparing command for rendering.

## License

This work is MIT-licensed.
The license applies to the pre-trained models as well.

## Citation

Please cite as 

```bibtex
@mastersthesis{mazlov21nrfaccel,
  author       = {Ilia Mazlov}, 
  title        = {Accelerations for Training and Rendering Neural Reflectance and Radiance Fields},
  school       = {University of Bonn},
  year         = 2021
}
```
