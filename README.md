# Alzheimer's Disease diagnosis based on Cauchy Convolutional Sparse Coding
by Perla Mayo, Robin Holmes and Alin M. Achim

The code contained in this repository corresponds to the implementation of the algorithm Cauchy Convolutional Sparse Coding (CCSC) for MRI classification for Alzheimer's Disease diagnosis.


## Implementation
The code has been implemented in Python and requires the following libraries to run:

* Pytorch
* Numpy
* Matplotlib
* SciKit
* SciPy


## How to run
To execute the program, you need to open a terminal where the code is located in your machine (**cd cauchycsc_dir**)

    python run_task.py -xml xml_path [-s seed] [-d seed]

The **required** arguments are:

 - xml_path The path of the XML containing all the settings for the execution. Examples of this XML file can be found in [architectures](architectures)

The **optional** arguments are:
- *seed* An initial seed to use. If none is specified, a random seed will be in place
- *dimension* The dimension of the input data to work with. This is not ideal and can be improved, but at this point this is still required

More details on the execution to come...
