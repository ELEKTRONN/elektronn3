Getting started with elektronn3
===============================

This guide describes how you can set up an environment for using elektronn3.
There are a lot of steps that you could do differently. We won't list all the
different options but just explain one way to get it running.

Installing Ananconda
--------------------

SSH to the server you want to run elektronn3 on and optionally open screen
so you can leave the SSH session at any time.

Example:

    $ ssh YOUR_USERNAME@example.com
    $ screen -rd || screen

Download Miniconda from https://docs.conda.io/en/latest/miniconda.html
(or download Anaconda).

Example:

    $ wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

Install it to `~/anaconda` by running

    $ bash Miniconda3-latest-Linux-x86_64.sh

following the assistant and keeping all defaults except the install location,
which you should switch to `YOUR_HOME_DIRECTORY/anaconda`.

Configuring conda
-----------------

After the installation is complete, create the file ~/.condarc and put the
following lines into it:

    channels:
      - pytorch
      - conda-forge
      - defaults
    pinned_packages:
      - python >=3.6
      - cudatoolkit =9.*  # Pin to a version that your GPU driver supports
      - blas >1.0  # Prevent conda from switching to mkl-blas 1.0
      - pytorch >=1.1.0

The restrictive pinned_packages configuration was at least necessary for me
because `conda update` sometimes does weird things and downgrades essential
packages. `pinned_packages` may not be required on your system.

Creating a conda environment for elektronn3
-------------------------------------------

Now you can create a new conda environment for elektronn3.

    $ cd
    $ git clone https://github.com/ELEKTRONN/elektronn3
    $ conda env create -f elektronn3/environment.yml

This should create a new conda environment called "elektronn3" with all the
necessary packages for using it.

**Note**: If you use other conda envs for different projects, the .condarc
above may cause conflicts due to its channel order and pinned_packges
restrictions.
If it causes problems, you can now move it to your dedicated env to limit its
effects to this env:

    mv ~/.condarc ~/anaconda/envs/elektronn3/

Note that elektronn3 itself is **not yet** installed because there is no conda
package available for it yet.

Activating the new environment
------------------------------

From now on, always make sure the new environment is active by running

    $ conda activate elektronn3

You will see an indicator in your shell that tells you you're in the right env.

**Note**: The environment activation only lasts for the current shell session.
You will need to `conda activate elektronn3` in every shell session in which
you want to use elektronn3.

Installing the elektronn3 package
---------------------------------

To install elektronn3 itself, you will have to use pip:

    $ pip install -e --no-deps ~/elektronn3

Explanation of the flags above:

- `--no-deps` prevents accidental installation of dependencies (this can
  sometimes happen because pip has problems with dependency resolution)
- `-e` makes pip install it in "editable" mode. This means that you can
  (and should) update your local elektronn3 package by navigating to
  its directory and pulling the latest changes:

      $ cd ~/elektronn3
      $ git pull

Next steps
----------

See https://github.com/ELEKTRONN/elektronn3/blob/master/README.md#training
