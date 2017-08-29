import PyZOGY.util as util
import mock_image_class
import numpy as np

def test_center_psf():
    psf = np.zeros((50, 51))
    psf[25, 25] = 1
    assert(util.center_psf(psf)[0, 0] == 1)


def test_make_mask():
    saturation = 1.
    image = mock_image_class.MockImageClass(shape=(50, 50), saturation=saturation)
    image[25, 25] = 2.
    image[30, 30] = 1.
    mask = util.make_mask(image, image.saturation, image.mask)
    assert(mask[image >= saturation].all() == True)


def test_interpolate_bad_pixels():
    image = mock_image_class.MockImageClass(shape=(50, 50))
    image[0, 25] = np.nan
    image[25, 0] = np.nan
    image[25, 25] = np.nan
    image.mask = np.zeros(shape=(50, 50))
    image.mask[np.isnan(image)] = 1
    image = util.interpolate_bad_pixels(image)
    assert(np.isfinite(image).all())

