# elektronn3

A PyTorch-based library that provides common functionality for 3D
convolutional neural networks, like data loading/augmentation,
training and model evaluation.

**It is currently in a very early stage of development** and will
undergo major breaking changes in the next weeks, so we don't
recommend using it yet if you are not already familiar with the code.


# Requirements

- Linux (support for Windows, MacOS and other systems is not planned)
- Python 3.6 or later
- PyTorch master 0.4.0 (unreleased, [7b33ef4](https://github.com/pytorch/pytorch/tree/7b33ef4cffed0dcd5c2506c4db1b2624736a22a3))
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

To test training with an extremely small model and **elektronn3**,
you can run:

    python3 scripts/train.py simple

You can replace `simple` by `n3d`, `vnet` or ``fcn32s`` to try other
integrated network models.

(Note: We will add support for supplying your own models in separate files
soon.)


## Training shell

- Hitting Ctrl-C anytime during the training will drop you to the
IPython training shell where you can access training data and make interactive
changes.
- To continue training, hit Ctrl-D twice.
- If you want the process to terminate after leaving the shell, set
`self.terminate = True` inside it and then hit Ctrl-D twice.


## Using Tensorboard

Tensorboard logs are saved in `~/tb/` by default, so you can track training
progress by running a tensorboard server there:

    tensorboard --logdir ~/tb

Then you can view the visualizations at http://localhost:6006.


# Contributors


The **elektronn3** project is being developed by the
[ELEKTRONN team](https://github.com/orgs/ELEKTRONN/people) at the
Max Planck Institute of Neurobiology and is funded by
[Winfried Denk's lab](http://www.neuro.mpg.de/denk).

[Jörgen Kornfeld](http://www.neuro.mpg.de/person/43611/3242677)
is academic advisor to this project.
