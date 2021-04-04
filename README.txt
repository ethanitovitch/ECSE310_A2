Python 3.9.0 was used to write and test our entire program.

# Setup

## Install dependencies

We use `argparse`, a module in the Python standard library, to parse the CLI arguments given to the fft.py script. The module included in the standard library is usually quite old, and does not support manually handling parsing errors which is required by our implementation  of the CLI application. Version 3.9 introduces an option to catch parsing errors which enables us to handle them in a more understandable and consistent way.

The libraries required by the assignment, namely `matplotlib`, `numpy`, and `opencv-python` do not come packaged with most Python installations and are not part of the standard library.

Thus our code requires several dependencies be installed (with specific versions), so we use (pip)[https://packaging.python.org/key_projects/#pip] for installing Python packages and (venv)[https://packaging.python.org/key_projects/#venv] for managing a virtual environments.

Alternative Python package installers, virtual environment managers, and overall dependency installations procedures could be used so long as they result in Python 3.9.0+, the Python standard library, and the required libraries at the versions specified in the `requirements.txt` file are all installed and available to the scripts.

The process to satisfy the dependencies in a CLI environment follows (this guide)[https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/], summarized below (assuming Python 3 and pip are installed).

First navigate to the root of the repository (which contains the `src/` and `test/` directories). Then follow the steps below.

1. Create the virtual environment

macOS and Linux:

`python3 -m venv env`

Windows:

`py -m venv env`

2. Activate the virtual environment

macOS and Linux:

`source env/bin/activate`

Windows:

`.\env\Scripts\activate`

3. Install required packages

macOS, Linux, and Windows:

`python3 -m pip install -r requirements.txt`

After satisfying the dependencies, you main invoke the program or run the tests as explained below.

# Invoke the program

From the command line, use the Python 3 interpreter you have installed on your machine and available in your virtual environment to run the `fft.py` script, located in the root directory. Provide the desired arguments after the script name according to the handout.

If your interpreter is in your shell PATH as `python3` and your present working directory is the `src/` directory in the repository, this would look like:

`python3 fft.py [-m mode] [-i image]`

e.g.

`python3 fft.py -m 4 -i 'moonlanding.png'`

### Usage

Display the CLI program usage with the `-h` or `--help` options, e.g.:

`python3 fft.py -h`


# Run the tests

From the command line, use the Python 3 interpreter you have installed on your machine and available in your virtual environment to run the `test.py` script, located in the root directory. This script does not read in any arguments.

If your interpreter is in your shell PATH as `python3` and your present working directory is the `src/` directory in the repository, this would look like:

`python3 test.py`
