import numpy as np
from astropy.io import fits
from PyZOGY import util


class ImageClass(np.ndarray):
    """Contains the image and relevant parameters"""

    def __new__(cls, image_filename, psf_filename, mask_filename=None, n_stamps=1,
                saturation=None, variance=None, read_noise=0, registration_noise=(0, 0)):
        raw_image, header = fits.getdata(image_filename, header=True)
        raw_psf = fits.getdata(psf_filename)
        psf = util.center_psf(util.resize_psf(raw_psf, raw_image.shape), fname=image_filename) / np.sum(raw_psf)
        if mask_filename is not None:
            mask = fits.getdata(mask_filename)
        else:
            mask = mask_filename
        mask = util.mask_saturated_pix(raw_image, saturation, mask, image_filename)
        masked_image = np.ma.array(raw_image, mask=mask)
        background_std, background_counts = util.fit_noise(masked_image, n_stamps=n_stamps, fname=image_filename)
        image_data = util.interpolate_bad_pixels(masked_image, fname=image_filename) - background_counts

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
        obj.psf = psf
        obj.zero_point = 1.
        obj.variance = variance
        obj.read_noise = read_noise
        obj.registration_noise = registration_noise

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
        self.read_noise = getattr(obj, 'read_noise', None)
        self.registration_noise = getattr(obj, 'registration_noise', None)
