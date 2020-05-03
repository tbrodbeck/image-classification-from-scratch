# Image Classification from Scratch
#### A Deep learning for Computer Vision Demo Environment by Tillmann Brodbeck

## imageClassificationFromScratch.ipynb

This is a demo environment containing all the necessary steps to install a dataset, install a deep learning model and analyze its performance, subsequently.
In general it was made sure that every critical step in the Python code is commented and "speaking" variables were used to make every bit of code understandable (and to avoid commenting every single line).

The first section is about setting up your environment from scratch. Here, a working python3 environment is a prerequisite. 
It is explained how to download an dataset with `wget`. Additionally all required python libraries are listed.

The second section demonstrates which hyperparameters are set and how different hyperparameter configurations can be managed. Additionally it shows how the data preprocessing is done and demonstrates two ways of installing a dataset for training: CIFAR10 is downloaded from `keras` directly and imagenette is imported from disk.

Section 3 demonstrates how the models are created and displays the exact architectures that are used in this demonstration in an easy way, built with as `Sequential` `keras` models. Additionally, it demonstrates how the training process can be initiated and outputs the `accuracy` metric and the categorical crossentropy as loss in real time while training.

Section 4 demonstrates how the model and its results can be saved to disk using `pickle`.

Finally, section 5 shows how trained models can be analyzed. Optionally, models can be imported from disk and tested against the dataset created in section 2. Finally it is shown how simple plots of the training and testing metrics can be created using `matplotlib.pyplot`.

## searchBestSetup.py and scheduleRuns.sh
These are scripts that help creating multiple runs from the command line to search for an optimal hyperparameter configuartions.

`searchBestSetup.py` contains the content of the `imageClassificationFromScratch.py` in the form of a python script that can be executed from the command line.

`scheduleRuns.sh` is a bash script to create multiple executions of `searchBestSetup.py` sequentially.

