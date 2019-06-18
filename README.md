# Multidimensional Contrast Limited Adaptive Histogram Equalization

## Introduction



## Sample datasets

### Fluorescence microscopy

Fluorescence microscopy can be used to capture time resolved volumetric images of a developing embryo. To illustrate the
effectiveness of MCLAHE we applied it to a dataset of an organism of species phallusia mammillata (available
[here](http://bioemergences.iscpif.fr/bioemergences/openworkflow-datasets.php)). To reduce the noise in the data, we
preprocessed it by a median filter with kernel size (2, 2, 2, 1) in the (x, y, z, t) space 

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![MCLAHE applied to fluorescence microscopy data](https://github.com/VincentStimper/mclahe/blob/master/images/demo_fm.gif "MCLAHE applied to fluorescence microscopy data")

The above image show a slice along the z-axis through the data. 4D CLAHE clearly enhances the contrast of the data.
Our hyperparameters of choice were the kernel size (20, 20, 10, 25), 256 bins in the histogram, and a clipping limit of
0.25. We used a global histogram range


### Multidimensional photoemission spectroscopy

Multidimensional photoemission spectroscopy is a technique to map the electronic band structure in a 4D space consisting
of two momentum (k<sub>x</sub>, k<sub>y</sub>), an energy (E), and a time (t) coordinate. In the raw data, the excited
state (E > 0) is not visible. 4D CLAHE makes is visible while enhancing local features in the other states but preserving
the temporal intensity changes.

![MCLAHE applied to MPES data](https://github.com/VincentStimper/mclahe/blob/master/images/demo_mpes.gif "MCLAHE applied to MPES data")

Here, we used a kernel size of (30, 30, 15, 20) in (k<sub>x</sub>, k<sub>y</sub>, E, t) space, 256 histogram bins, and
a clipping limit of 0.02. We made use of the adaptive histogram range for processing this dataset.
