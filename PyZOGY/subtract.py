from . import util
from astropy.io import fits
import astropy
from distutils.version import LooseVersion
import numpy as np


class ImageClass:
    """Contains the image and relevant parameters"""

    def __init__(self, image_filename, psf_filename, mask_filename='', n_stamps=1, saturation=np.inf, variance = np.inf):
        self.image_filename = image_filename
        self.psf_filename = psf_filename

        self.raw_image_data = fits.getdata(image_filename)
        self.raw_psf_data = fits.getdata(psf_filename)

        self.saturation = saturation

        if mask_filename == '':
            self.pixel_mask = util.make_pixel_mask(self.raw_image_data, self.saturation)
        else:
            self.pixel_mask = util.make_pixel_mask(self.raw_image_data, self.saturation, fits.getdata(mask_filename))
        self.masked_image_data = np.ma.array(self.raw_image_data, mask=self.pixel_mask)

        self.psf_data = util.center_psf(util.resize_psf(self.raw_psf_data, self.raw_image_data.shape))
        self.psf_data /= np.sum(self.raw_psf_data)

        self.zero_point = 1.
        self.variance = variance
        self.background_std, self.background_counts = util.fit_noise(self.masked_image_data, n_stamps=n_stamps)
        self.image_data = util.interpolate_bad_pixels(self.masked_image_data) - self.background_counts


def calculate_difference_image(science, reference,
                               normalization='reference', output='output.fits', gain_ratio=np.inf):
    """Calculate the difference image using the Zackay algorithm"""

    # match the gains
    if gain_ratio == np.inf:
        science.zero_point = util.solve_iteratively(science, reference)
        zero_point_ratio = science.zero_point / reference.zero_point
    else:
        zero_point_ratio = gain_ratio

    # create required arrays
    science_image = science.image_data
    reference_image = reference.image_data
    science_psf = science.psf_data
    reference_psf = reference.psf_data

    # do fourier transforms (fft)
    science_image_fft = np.fft.fft2(science_image)
    reference_image_fft = np.fft.fft2(reference_image)
    science_psf_fft = np.fft.fft2(science_psf)
    reference_psf_fft = np.fft.fft2(reference_psf)

    # calculate difference image
    denominator = science.background_std ** 2 * abs(reference_psf_fft) ** 2
    denominator += reference.background_std ** 2 * zero_point_ratio ** 2 * abs(science_psf_fft) ** 2
    difference_image_fft = science_image_fft * reference_psf_fft
    difference_image_fft -= zero_point_ratio * reference_image_fft * science_psf_fft
    difference_image_fft /= np.sqrt(denominator)
    difference_image = np.fft.ifft2(difference_image_fft)
    difference_image = normalize_difference_image(difference_image, science, reference, normalization=normalization)
    save_difference_image_to_file(difference_image, science, normalization, output)

    return difference_image


def calculate_difference_image_zero_point(science, reference):
    """Calculate the flux based zero point of the difference image"""

    zero_point_ratio = science.zero_point / reference.zero_point
    denominator = science.background_std ** 2 + reference.background_std ** 2 * zero_point_ratio ** 2
    difference_image_zero_point = zero_point_ratio / np.sqrt(denominator)

    return difference_image_zero_point

def calculate_difference_psf(science, reference):
    """Calculate the psf of the difference image"""

    science_psf_fft = np.fft.fft2(science.psf_data)
    reference_psf_fft = np.fft.fft2(reference.psf_data)
    denominator = science.background_std ** 2 * abs(reference_psf_fft) ** 2
    denominator += reference.background_std ** 2 * science.zero_point ** 2 * abs(science_psf_fft) ** 2
    denominator *= calculate_difference_image_zero_point(science, reference)

    difference_psf_fft = science.zero_point * science_psf_fft * reference_psf_fft
    difference_psf_fft /= np.sqrt(denominator)
    difference_psf = np.fft.ifft2(difference_psf_fft)
    return difference_psf


def calculate_matched_filter_image(science, reference, photometry=True, normalization = 'None'):
    """Calculate the matched filter difference image"""

    difference_image = calculate_difference_image(science, reference, normalization=normalization)
    difference_psf = calculate_difference_psf(science, reference)
    difference_zero_point = calculate_difference_image_zero_point(science, reference)
    matched_filter_fft = difference_zero_point * np.fft.fft2(difference_image) * np.conj(np.fft.fft2(difference_psf))
    matched_filter = np.fft.ifft2(matched_filter_fft)

    if (science.variance != np.inf) and (reference.variance != np.inf):
        # add variance correction here
        matched_filter /= 1

    if photometry:
        science_psf_fft = np.fft.fft2(science.psf_data)
        reference_psf_fft = np.fft.fft2(reference.psf_data)
        zero_point = science.zero_point ** 2 * reference.zero_point ** 2
        zero_point *= abs(science_psf_fft) ** 2 * abs(reference_psf_fft) ** 2
        denominator = reference.background_std ** 2 * science.zero_point ** 2 * abs(science_psf_fft) ** 2
        denominator += science.background_std ** 2 * reference.zero_point ** 2 * abs(reference_psf_fft) ** 2
        zero_point /= denominator
        matched_filter /= np.sum(zero_point)

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


def run_subtraction(science_image, reference_image, science_psf, reference_psf, output = 'output.fits',
                    science_mask = '', reference_mask = '', n_stamps = 1, normalization = 'reference',
                    science_saturation = False, reference_saturation = False, science_variance=np.inf,
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
