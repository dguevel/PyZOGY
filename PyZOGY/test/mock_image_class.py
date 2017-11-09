import numpy as np
from astropy.io import fits

class MockImageClass(np.ndarray):
    """Creates a mock version of ImageClass for testing"""

    def __new__(cls, image_filename='', psf_filename='', mask_filename=None, n_stamps=1, saturation=np.inf, variance=np.inf, shape=(50,50)):
        raw_image, header = np.ones(shape), fits.Header()#fits.getdata(image_filename, header=True)
        raw_psf = np.ones(shape)
        mask = np.zeros(shape)
        background_std, background_counts = np.ones(shape), np.zeros(shape)
        image_data = np.ones(shape)

        obj = np.asarray(image_data).view(cls)
        obj.header = header
        obj.raw_image = raw_image
        obj.raw_psf = raw_psf
        obj.background_std = background_std
        obj.background_counts = background_counts
        obj.image_filename = image_filename
        obj.psf_filename = psf_filename
        obj.saturation = saturation
        obj.mask = mask
        obj.psf = raw_psf
        obj.zero_point = 1.
        obj.variance = variance

        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.raw_image = getattr(obj, 'raw_image', None)
        self.header = getattr(obj, 'header', None)
        self.raw_psf = getattr(obj, 'raw_psf', None)
        self.background_std = getattr(obj, 'background_std', None)
        self.background_counts = getattr(obj, 'background_counts', None)
        self.image_filename = getattr(obj, 'image_filename', None)
        self.psf_filename = getattr(obj, 'psf_filename', None)
        self.saturation = getattr(obj, 'saturation', None)
        self.mask = getattr(obj, 'mask', None)
        self.psf = getattr(obj, 'psf', None)
        self.zero_point = getattr(obj, 'zero_point', None)
        self.variance = getattr(obj, 'variance', None)
