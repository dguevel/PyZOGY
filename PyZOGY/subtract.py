from . import util
from astropy.io import fits
import astropy
from distutils.version import LooseVersion
import numpy as np


class ImageClass(np.ndarray):
    """Contains the image and relevant parameters"""

    def __new__(cls, image_filename, psf_filename, mask_filename='', n_stamps=1, saturation=np.inf, variance=np.inf):
        raw_image = fits.getdata(image_filename)
        raw_psf = fits.getdata(psf_filename)
        mask = util.make_mask(raw_image, saturation, mask_filename)
        background_std, background_counts = util.fit_noise(raw_image, n_stamps=n_stamps, output_name=image_filename)
        image_data = util.interpolate_bad_pixels(raw_image, mask) - background_counts

        obj = np.asarray(image_data).view(cls)
        obj.raw_image = raw_image
        obj.raw_psf = raw_psf
        obj.background_std = background_std
        obj.background_counts = background_counts
        obj.image_filename = image_filename
        obj.psf_filename = psf_filename
        obj.saturation = saturation
        obj.mask = mask
        obj.psf = util.center_psf(util.resize_psf(raw_psf, raw_image.shape)) / np.sum(raw_psf)
        obj.zero_point = 1.
        obj.variance = variance

        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.raw_image = getattr(obj, 'raw_image', None)
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


def calculate_difference_image(science, reference,
                               normalization='reference', output='output.fits', gain_ratio=np.inf):
    """Calculate the difference image using the Zackey algorithm"""

    # match the gains
    if gain_ratio == np.inf:
        science.zero_point = util.solve_iteratively(science, reference)

    # create required arrays
    science_image = science
    reference_image = reference
    science_psf = science.psf
    reference_psf = reference.psf

    # do fourier transforms (fft)
    science_image_fft = np.fft.fft2(science_image)
    reference_image_fft = np.fft.fft2(reference_image)
    science_psf_fft = np.fft.fft2(science_psf)
    reference_psf_fft = np.fft.fft2(reference_psf)

    # calculate difference image
    denominator = science.background_std ** 2 * reference.zero_point ** 2 * abs(reference_psf_fft) ** 2
    denominator += reference.background_std ** 2 * science.zero_point ** 2 * abs(science_psf_fft) ** 2
    difference_image_fft = science_image_fft * reference_psf_fft * reference.zero_point
    difference_image_fft -= reference_image_fft * science_psf_fft * science.zero_point
    difference_image_fft /= np.sqrt(denominator)
    difference_image = np.fft.ifft2(difference_image_fft)
    difference_image = normalize_difference_image(difference_image, science, reference, normalization=normalization)
    save_difference_image_to_file(difference_image, science, normalization, output)

    return difference_image


def calculate_difference_image_zero_point(science, reference):
    """Calculate the flux based zero point of the difference image"""

    denominator = science.background_std ** 2 * reference.zero_point ** 2
    denominator += reference.background_std ** 2 * science.zero_point ** 2
    difference_image_zero_point = science.zero_point * reference.zero_point / np.sqrt(denominator)

    return difference_image_zero_point


def calculate_difference_psf(science, reference):
    """Calculate the psf of the difference image"""

    science_psf_fft = np.fft.fft2(science.psf)
    reference_psf_fft = np.fft.fft2(reference.psf)
    denominator = science.background_std ** 2 * reference.zero_point ** 2 * abs(reference_psf_fft) ** 2
    denominator += reference.background_std ** 2 * science.zero_point ** 2 * abs(science_psf_fft) ** 2
    difference_zero_point = calculate_difference_image_zero_point(science, reference)

    difference_psf_fft = science.zero_point * science_psf_fft * reference_psf_fft
    difference_psf_fft /= (difference_zero_point * np.sqrt(denominator))
    difference_psf = np.fft.ifft2(difference_psf_fft)
    return difference_psf


def calculate_matched_filter_image(science, reference, photometry=True, normalization='none'):
    """Calculate the matched filter difference image"""

    difference_image = calculate_difference_image(science, reference, normalization='none')
    difference_psf = calculate_difference_psf(science, reference)
    difference_zero_point = calculate_difference_image_zero_point(science, reference)
    matched_filter_fft = difference_zero_point * np.fft.fft2(difference_image) * np.conj(np.fft.fft2(difference_psf))
    matched_filter = np.fft.ifft2(matched_filter_fft)

    if (science.variance != np.inf) and (reference.variance != np.inf):
        # add variance correction here
        matched_filter /= 1

    if photometry:
        matched_filter = calculate_photometry(matched_filter, science, reference, normalization=normalization)

    return matched_filter


def normalize_difference_image(difference, science, reference, normalization='reference'):
    """Normalize to user's choice of image"""

    difference_image_zero_point = calculate_difference_image_zero_point(science, reference)
    if normalization == 'reference' or normalization == 't':
        difference_image = difference * reference.zero_point / difference_image_zero_point
    elif normalization == 'science' or normalization == 'i':
        difference_image = difference * science.zero_point / difference_image_zero_point
    else:
        difference_image = difference

    return difference_image


def calculate_photometry(matched_filter, science, reference, normalization):
    """Calculate the photometry from the matched filter image"""

    science_psf_fft = np.fft.fft2(science.psf)
    reference_psf_fft = np.fft.fft2(reference.psf)
    zero_point = science.zero_point ** 2 * reference.zero_point ** 2
    zero_point *= abs(science_psf_fft) ** 2 * abs(reference_psf_fft) ** 2
    denominator = reference.background_std ** 2 * science.zero_point ** 2 * abs(science_psf_fft) ** 2
    denominator += science.background_std ** 2 * reference.zero_point ** 2 * abs(reference_psf_fft) ** 2
    zero_point /= denominator
    matched_filter /= np.sum(zero_point)
    print(np.sum(zero_point))
    if normalization == 'science':
        matched_filter = matched_filter * science.zero_point
    else:
        matched_filter = matched_filter * reference.zero_point
    return matched_filter




def run_subtraction(science_image, reference_image, science_psf, reference_psf, output='output.fits',
                    science_mask='', reference_mask='', n_stamps=1, normalization='reference',
                    science_saturation=np.inf, reference_saturation=np.inf, science_variance=np.inf,
                    reference_variance=np.inf, matched_filter=False, photometry=True, gain_ratio=np.inf):
    """Run full subtraction given filenames and parameters"""

    science = ImageClass(science_image, science_psf, science_mask, n_stamps, science_saturation)
    reference = ImageClass(reference_image, reference_psf, reference_mask, n_stamps, reference_saturation)
    if matched_filter:
        difference=calculate_matched_filter_image(science, reference, normalization=normalization, photometry=photometry)
    else:
        difference = calculate_difference_image(science, reference, normalization, output, gain_ratio=gain_ratio)
    save_difference_image_to_file(difference, science, normalization, output)


def save_difference_image_to_file(difference_image, science, normalization, output):
    """Save difference image to file"""

    hdu = fits.PrimaryHDU(np.real(difference_image))
    hdu.header = fits.getheader(science.image_filename)
    hdu.header['PHOTNORM'] = normalization
    hdu.header['CONVOL00'] = normalization

    # clobber keyword is deprecated in astropy 1.3
    if LooseVersion(astropy.__version__) < LooseVersion('1.3'):
        hdu.writeto(output, clobber=True, output_verify='warn')
    else:
        hdu.writeto(output, overwrite=True, output_verify='warn')
