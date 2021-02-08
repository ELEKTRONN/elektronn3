[![Read the Docs](https://readthedocs.org/projects/elektronn3/badge/?version=latest)](https://elektronn3.readthedocs.io/en/latest/)

# elektronn3

A PyTorch-based library for working with 3D and 2D convolutional neural networks, with focus on semantic segmentation of volumetric biomedical image data.

Quick overview of **elektronn3**'s code structure:

- **`elektronn3.training`**: Utilities for training, monitoring, visualization and model evaluation. Provides a flexible `Trainer` class that can be used for arbitrary PyTorch models and Data sets.
- **`elektronn3.data`**: Data loading and augmentation code for semantic segmentation and other dense prediction tasks. The main focus is on 3D (volumetric) biomedical image data stored as HDF5 files, but most of the code also supports 2D and n-dimensional data.
- **`elektronn3.inference`**: Code for deployment of trained models and for efficient tiled inference on large input volumes.
- **`elektronn3.models`**: Neural network architectures for segmentation and other pixel-wise prediction tasks. `models.unet.UNet` provides a highly flexible PyTorch model class inspired by [3D U-Net](https://arxiv.org/abs/1606.06650) that works in 2D and 3D and supports custom depths, data anisotropy handling, batch normalization and many more configurable features.
- **`elektronn3.modules`**: Modules (in the sense of `torch.nn.Module`) for building neural networks and loss functions.
- **`examples`**: Scripts that demonstrate how the library can be used for biomedical image segmentation.

**elektronn3**'s modular codebase makes it easy to extend/replace parts of it with your own code: For example, you can use the training tools included in `elektronn3.training` with your own data sets, augmentation methods, network models etc. or use the data loading and augmentation code of `elektronn3.data` with your own training code. The neural network architectures in `elektronn3.models` can also be freely used with custom training and/or data loading code.

Documentation can be found at [elektronn3.readthedocs.io](https://elektronn3.readthedocs.io).

For a roadmap of planned features, see the ["enhancement" issues on the tracker](https://github.com/ELEKTRONN/elektronn3/issues?q=is%3Aissue+is%3Aopen+label%3Aenhancement).

# Requirements

- Linux (support for Windows, MacOS and other systems is not planned)
- Python 3.6 or later
- PyTorch 1.5 or later (earlier versions may work, but are untested)
- For other requirements see `requirements.txt`

# Setup

Ensure that all of the requirements listed above are installed.
We recommend using conda or a virtualenv for that.
To install **elektronn3** in development mode, run

    git clone https://github.com/ELEKTRONN/elektronn3 elektronn3-dev
    pip install -e elektronn3-dev

To update your installation, just `git pull` in your clone
directory.

If you are not familiar with virtualenv and conda or are not sure about some of
the required steps, you can find a more detailed setup guide [here](https://github.com/ELEKTRONN/elektronn3/blob/master/setup.md)

# Training

For a quick test run, first ensure that the neuro_data_cdhw data set is
in the expected path:

    wget https://github.com/ELEKTRONN/elektronn.github.io/releases/download/neuro_data_cdhw/neuro_data_cdhw.zip
    unzip neuro_data_cdhw.zip -d ~/neuro_data_cdhw

To test training with our custom U-Net-inspired architecture in **elektronn3**,
you can run:

    python3 train_unet_neurodata.py


## Using Tensorboard

Tensorboard logs are saved in `~/e3training/` by default, so you can track training
progress by running a tensorboard server there:

    tensorboard --logdir ~/e3training/

Then you can view the visualizations at http://localhost:6006.

# Contributors

The **elektronn3** project is being developed by the
[ELEKTRONN team](https://github.com/orgs/ELEKTRONN/people).
[Jörgen Kornfeld](https://twitter.com/jmrko) is academic advisor to this project.
