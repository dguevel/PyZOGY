import PyZOGY.util as util
import numpy as np

def test_center_psf():
    test_psf = np.zeros((3, 3))
    test_psf[1,1] = 1.
    assert(util.center_psf(test_psf)[0,0] == 1)

