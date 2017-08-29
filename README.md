# PyZOGY
PyZOGY is a Python implementation of the image subtraction algorithm published by Zackay, Ofek, and Gal-Yam. 
The algorithm requires two registered images and their PSF's saved as fits files. One can optionally provide
masks, in fits files where every pixel is either 0 (good) or 1 (bad). Alternatively, the code will mask pixels
above a user defined threshold (a number) for each image. The code fits the spatially varying background level by dividing 
the image into a number of stamps provided by the user; the default is 1. The image can be normalized to either
the science image or the reference image.


The details of the algorithm can be found at http://iopscience.iop.org/article/10.3847/0004-637X/830/1/27/meta

## Installation
Clone the repository and run `python setup.py install`

## Usage
The code can be run from the command line or within Python

To run on the command line, type:

`pyzogy --science-image "your-science-image" --reference-image "your-reference-image" --science-psf "your-science-psf" --reference-psf "your-reference-psf"`

with any of the following options:

```
--science-mask "your-science-mask"
--reference-mask "your-reference-mask"
--science-saturation number
--reference-saturation number
--n-stamps "number"
--normalization "science" or "reference"
--gain-ratio number
--gain-mask "mask-filename"
--use-pixels
--show
--matched-filter "your-matched-filter-output"
```
To use in Python, type:
```
from PyZOGY.subtract import run_subtraction
run_subtraction("your-science-image", "your-reference-image", "your-science-psf", "your-reference-psf")
```

with any of the following options:

```
science_mask = "your-science-mask"
reference_mask = "youre-reference-mask"
science_saturation = number
reference_saturation = number
n_stamps = number
normalization = "science" or "reference"
gain_ratio = number
gain-mask = "mask-filename"
use-pixels = boolean
show = boolean
matched-filter = "your-matched-filter-output"
```

## Dependencies

PyZOGY requires numpy, astropy, scipy, sep, matplotlib, and statsmodels
