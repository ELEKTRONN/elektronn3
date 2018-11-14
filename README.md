[![Read the Docs](https://readthedocs.org/projects/elektronn3/badge/?version=latest)](https://elektronn3.readthedocs.io/en/latest/)

# elektronn3

A PyTorch-based library for research on
convolutional neural networks for 3D semantic segmentation.
Its focus is on HDF5 data loading/augmentation, training, monitoring
and model evaluation.

**It is currently in a very early stage of development** and will
undergo major breaking changes in the next weeks, so we don't
recommend using it yet if you are not already familiar with the code.

For a roadmap of planned features, see the ["enhancement" issues on the tracker](https://github.com/ELEKTRONN/elektronn3/issues?q=is%3Aissue+is%3Aopen+label%3Aenhancement).

# Requirements

- Linux (support for Windows, MacOS and other systems is not planned)
- Python 3.6 or later
- PyTorch 0.4.1 or a recent nightly version (1.0.0 preview)
- For other requirements see `requirements.txt`


# Setup

Ensure that all of the requirements listed above are installed.
We recommend using conda or a virtualenv for that.
To install **elektronn3** in development mode, run

    git clone https://github.com/ELEKTRONN/elektronn3 elektronn3-dev
    pip install -e elektronn3-dev

To update your installation, just `git pull` in your clone
directory.

# Training

For a quick test run, first ensure that the neuro_data_cdhw data set is
in the expected path:

    wget https://github.com/ELEKTRONN/elektronn.github.io/releases/download/neuro_data_cdhw/neuro_data_cdhw.zip
    unzip neuro_data_cdhw.zip -d ~/neuro_data_cdhw

To test training with our custom U-Net-derived architecture in **elektronn3**,
you can run:

    python3 train_unet_neurodata.py


## Training shell

- Hitting Ctrl-C anytime during the training will drop you to the
IPython training shell where you can access training data and make interactive
changes.
- To continue training, hit Ctrl-D twice.
- If you want the process to terminate after leaving the shell, set
`self.terminate = True` inside it and then hit Ctrl-D twice.


## Using Tensorboard

Tensorboard logs are saved in `~/e3training/` by default, so you can track training
progress by running a tensorboard server there:

    tensorboard --logdir ~/e3training/

Then you can view the visualizations at http://localhost:6006.


# Contributors


The **elektronn3** project is being developed by the
[ELEKTRONN team](https://github.com/orgs/ELEKTRONN/people) at the
Max Planck Institute of Neurobiology and is funded by
[Winfried Denk's lab](http://www.neuro.mpg.de/denk).

[Jörgen Kornfeld](http://www.neuro.mpg.de/person/43611/3242677)
is academic advisor to this project.
