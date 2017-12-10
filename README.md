# elektronn3

A PyTorch-based library that provides common functionality for 3D
convolutional neural networks, like data loading/augmentation,
training and model evaluation.

**It is currently in a very early stage of development** and will
undergo major breaking changes in the next weeks, so we don't
recommend using it yet if you are not already familiar with the code.


# Requirements

- Python 3.6 or later
- For other requirements see `requirements.txt`


# Setup


For a quick local test run, first ensure that the neuro_data_zxy data set is
in the expected path:

    $ wget http://elektronn.org/downloads/neuro_data_zxy.zip
    $ unzip neuro_data_zxy.zip -d ~/neuro_data_zxy


# Training
To simply test training with an extremely small model and **elektronn3**,
you can run:

    $ python3 train.py simple --host local


## Training shell

- Hitting Ctrl-C anytime during the training will drop you to the
IPython training shell where you can access training data and make interactive
changes.
- To continue training, hit Ctrl-D twice.
- If you want the process to terminate after leaving the shell, set
`self.terminate = True` inside it and then hit Ctrl-D twice.


## Options

- You can replace `simple` by `n3d`, `vnet` or ``fcn32s`` to try other network
  models.
- `--host` can also be set to `wb` if you run it on the wb server.
  Then it will use the data set at `/wholebrain/scratch/j0126/barrier_gt_phil/`
  (This setting will is just for quick lab-internal testing and will be removed later)


## Using Tensorboard

Tensorboard logs are saved in `~/tb/` by default, so you can track training
progress by running a tensorboard server there:

    $ tensorboard --logdir ~/tb

Then you can view the visualizations at http://localhost:6006.


# Contributors


The **elektronn3** project is being developed by the
[ELEKTRONN team](https://github.com/orgs/ELEKTRONN/people) at the
Max Planck Institute of Neurobiology and is funded by
[Winfried Denk's lab](http://www.neuro.mpg.de/denk).

`Jörgen Kornfeld <http://www.neuro.mpg.de/person/43611/3242677>`_
is academic advisor to this project.
