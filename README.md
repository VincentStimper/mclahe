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


## Documentation

```python
def mclahe(x, kernel_size=None, n_bins=128, clip_limit=0.01, adaptive_hist_range=False, use_gpu=True):
    ...
```

### Parameters

* `x`: Input data as a numpy array with a arbitrary number of dimensions
* `kernel_size`: Tuple, list, or numpy array specifying the kernel size along the data dimensions. If `kernel_size=None`,
the kernel size is set to 1/8 of the data size along each dimension. This is a typical choice for photographs. For more
complex dataset, the kernel size should be roughly of the size of the features which shall be enhanced by MCLAHE.
* `n_bins`: Integer specifying the number of histogram bins used within each kernel. Typically, it is set to a power of
two like 128 or 256 but any number could be chosen.
* `clip_limit`: Float being the share of voxels within a kernel at which the histogram shall be clipped. A clipping
limit of 1 corresponds to standard histogram equalization and if it is smaller than 1 the contrast enhancement is
limited. Since a relative histogram height of `1/n_bins` corresponds to a uniform distribution the clipping limit should
be higher than that.
* `adaptive_hist_range`: flag saying whether an adaptive histogram range (AHR) shall be used or not. With AHR, each
histogram uses its own range determined by the minimum and maximum intensity in the kernel. Otherwise, the range is set
by the global minimum and maximum within the data.
* `use_gpu`: flag specifying whether a GPU shall be used for computations if available. If there is no GPU available,
this flag has no influence. If the dataset is very large it might be necessary to use CPUs only to not run out of memory
on the GPU.


## Citation

If you are using this package within your own projects, please cite it as
> Vincent Stimper, Stefan Bauer, Ralph Ernstorfer, Bernhard Schölkopf, and R. Patrick Xian: Multidimensional Contrast
> Limited Adaptive Histogram Equalization, arXiv: 1906.11355, 2019.

Bibtex code
```
@Misc{Stimper2019,
  author = {Stimper, Vincent and Bauer, Stefan and Ernstorfer, Ralph
            and Sch{\"o}lkop, Bernhard and Xian, R. Patrick},
  title  = {Multidimensional Contrast Limited Adaptive Histogram Equalization},
  year   = {2019},
  eprint = {arXiv: 1906.11355},
}
```