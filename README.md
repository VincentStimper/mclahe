# Multidimensional Contrast Limited Adaptive Histogram Equalization

## Introduction

Multidimensional Contrast Limited Adaptive Histogram Equalization (MCLAHE) is a multidimensional extension of the
contrast enhancement procedure CLAHE for images. It can be applied to datasets with an arbitrary number of dimensions.
It is implemented in Tensorflow. Hence, it can be run on multiple CPUs or other hardware accelerators such as GPUs.


## Installation

You can install `mclahe` directly using pip:
```
pip install --upgrade git+https://github.com/VincentStimper/mclahe.git
```
Alternatively, you can first download or clone the repository on your computer via
```
git clone https://github.com/VincentStimper/mclahe.git
```
then navigate into the folder and install it via
```
pip install --upgrade .
```
or 
```
python setup.py install
```

### Requirements

The main package requires `numpy` and `tensorflow`. `tensorflow` needs to be installed manually depending on the hardware
in use. A comprehensive installation guide is given at the [Tensorflow webpage](https://www.tensorflow.org/install).
For the sample notebook, `matplotlib` is required as well. 

## Sample datasets

### Fluorescence microscopy

Fluorescence microscopy can be used to capture time resolved volumetric images of a developing embryo. To illustrate the
effectiveness of MCLAHE we applied it to a dataset of an organism of species phallusia mammillata (available
[here](http://bioemergences.iscpif.fr/bioemergences/openworkflow-datasets.php)). To reduce the noise in the data, we
preprocessed it by a median filter with kernel size (2, 2, 2, 1) in the (x, y, z, t) space 

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![MCLAHE applied to fluorescence microscopy data](https://github.com/VincentStimper/mclahe/blob/master/images/demo_fm.gif "MCLAHE applied to fluorescence microscopy data")

The above image show a slice along the z-axis through the data. The unit hpf means hours post fertilization. Our
hyperparameters of choice were the kernel size (20, 20, 10, 25), 256 bins in the histogram, and a clipping limit of
0.25. We used a global histogram range


### Multidimensional photoemission spectroscopy

Multidimensional photoemission spectroscopy is a technique to map the electronic band structure in a 4D space consisting
of two momentum (k<sub>x</sub>, k<sub>y</sub>), an energy (E), and a pump-probe time delay (t<sub>pp</sub>) coordinate. In the raw data, the excited
state (E > 0) is not visible. 4D CLAHE makes is visible while enhancing local features in the other states but preserving
the temporal intensity changes.

![MCLAHE applied to MPES data](https://github.com/VincentStimper/mclahe/blob/master/images/demo_mpes.gif "MCLAHE applied to MPES data")

Here, we used a kernel size of (30, 30, 15, 20) in (k<sub>x</sub>, k<sub>y</sub>, E, t<sub>pp</sub>) space, 256 histogram bins, and
a clipping limit of 0.02. We made use of the adaptive histogram range for processing this dataset.
