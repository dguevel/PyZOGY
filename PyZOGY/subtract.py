from . import util
from .image_class import ImageClass
from astropy.io import fits
import numpy as np
import logging

# clobber keyword is deprecated in astropy 1.3
from astropy import __version__
if __version__ < '1.3':
    overwrite = {'clobber': True}
else:
    overwrite = {'overwrite': True}


def calculate_difference_image(science, reference, gain_ratio=np.inf, gain_mask=None, sigma_cut=5., use_pixels=False,
                               show=False, percent=99, use_mask_for_gain=True, max_iterations=5, size_cut=True,
                               pixstack_limit=None):
    """
    Calculate the difference image using the Zackay algorithm.

    This is the main function that calculates the difference image using the
    Zackay, Ofek, Gal-Yam 2016. It operates on ImageClass objects defined in
    image_class.py. The function will fit the gain ratio if not provided.
    Ultimately this calculates equation 13 in Zackay, Ofek, Gal-Yam 2016.

    Parameters
    ----------
    science : PyZOGY.ImageClass
        ImageClass instance created from the science image.
    reference : PyZOGY.ImageClass
        ImageClass instance created from the reference image.
    gain_ratio : float, optional
        Ration of the gains or flux based zero points of the two images.
    gain_mask : str or numpy.ndarray, optional
        Array or FITS file holding an array of pixels to use when fitting
        the gain ratio.
    sigma_cut : float, optional
        Threshold (in standard deviations) to extract a star from the image (`thresh` in `sep.extract`).
    use_pixels : bool, optional
        Fit the gain ratio using pixels (True) or stars (False) in image.
    show : bool, optional
        Display debuggin plots during fitting.
    percent : float, optional
        Percentile cutoff to use for fitting the gain ratio.
    use_mask_for_gain : bool, optional
        Set to False in order to ignore the input masks when calculating the gain ratio.
    max_iterations : int, optional
        Maximum number of iterations to reconvolve the images for gain matching.
    size_cut : bool, optinal
        Ignore unusually large/small sources for gain matching (assumes most sources are real).
    pixstack_limit : int, optional
        Number of active object pixels in Sep, set with sep.set_extract_pixstack

    Returns
    -------
    difference_image : numpy.ndarray
        The difference between science and reference images.
    """

    # match the gains
    if gain_ratio == np.inf:
        if gain_mask is not None:
            if type(gain_mask) == str:
                gain_mask_data = fits.getdata(gain_mask)
            else:
                gain_mask_data = gain_mask
            science.mask[gain_mask_data == 1] = 1
            reference.mask[gain_mask_data == 1] = 1
        science.zero_point = util.solve_iteratively(science, reference, sigma_cut=sigma_cut, use_pixels=use_pixels,
                                                    show=show, percent=percent, use_mask=use_mask_for_gain,
                                                    max_iterations=max_iterations, size_cut=size_cut,
                                                    pixstack_limit=pixstack_limit)
    else:
        science.zero_point = gain_ratio

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
    difference_image = np.real(difference_image)

    return difference_image


def calculate_difference_image_zero_point(science, reference):
    """
    Calculate the flux based zero point of the difference image.
    
    Calculate the difference image flux based zero point using equation 15 of 
    Zackay, Ofek, Gal-Yam 2016.

    Parameters
    ----------
    science : PyZOGY.ImageClass
        ImageClass instance created from the science image.
    reference : PyZOGY.ImageClass
        ImageClass instance created from the reference image.

    Returns
    -------
    difference_image_zero_point : float
        Flux based zero point of the difference image.
    """

    denominator = science.background_std ** 2 * reference.zero_point ** 2
    denominator += reference.background_std ** 2 * science.zero_point ** 2
    difference_image_zero_point = science.zero_point * reference.zero_point / np.sqrt(denominator)

    logging.info('Global difference image zero point is {}'.format(np.mean(difference_image_zero_point)))
    return difference_image_zero_point


def calculate_difference_psf(science, reference, difference_image_zero_point):
    """
    Calculate the PSF of the difference image.
    
    Calculactes the PSF of the difference image using equation 17 of Zackay,
    Ofek, Gal-Yam 2016.

    Parameters
    ----------
    science : PyZOGY.ImageClass
        ImageClass instance created from the science image.
    reference : PyZOGY.ImageClass
        ImageClass instance created from the reference image.
    difference_image_zero_point : float
        Flux based zero point of the difference image.

    Returns
    -------
    difference_psf : numpy.ndarray
        PSF of the difference image.
    """

    science_psf_fft = np.fft.fft2(science.psf)
    reference_psf_fft = np.fft.fft2(reference.psf)
    denominator = science.background_std ** 2 * reference.zero_point ** 2 * abs(reference_psf_fft) ** 2
    denominator += reference.background_std ** 2 * science.zero_point ** 2 * abs(science_psf_fft) ** 2

    difference_psf_fft = science.zero_point * science_psf_fft * reference_psf_fft
    difference_psf_fft /= difference_image_zero_point * np.sqrt(denominator)
    difference_psf = np.fft.ifft2(difference_psf_fft)

    return difference_psf


def calculate_matched_filter_image(difference_image, difference_psf, difference_zero_point):
    """
    Calculate the matched filter difference image.
    
    Calculates the matched filter difference image described in Zackay, Ofek, 
    Gal-Yam 2016 defined in equation 16.

    Parameters
    ----------
    difference_image : numpy.ndarray
        A difference image as calculated using calculate_difference_image.
    difference_psf : numpy.ndarray
        PSF for the difference image above.
    difference_zero_point
        Flux based zero point for the image above.

    Returns
    -------
    matched_filter : numpy.ndarray
        Matched filter image.
    """

    matched_filter_fft = difference_zero_point * np.fft.fft2(difference_image) * np.conj(np.fft.fft2(difference_psf))
    matched_filter = np.fft.ifft2(matched_filter_fft)

    return matched_filter


def source_noise(image, kernel):
    """
    Calculate source noise correction for matched filter image
    
    Calculate the noise due to the sources in an image. The output is used by
    noise corrected matched filter image. This is equation 26 in Zackay, Ofek,
    Gal-Yam 2016.

    Parameters
    ----------
    image : PyZOGY.ImageClass
        ImageClass instance with read_noise attribute defined.
    kernel : numpy.ndarray
        Convolution kernel for the noise image. This comes from the function
        called noise_kernels.

    Returns
    -------
    image_variance_corr : numpy.ndarray
        Variance of the image due to source noise.
    """

    if image.variance is None:
        image.variance = np.copy(image.raw_image) + image.read_noise

    image_variance_corr = np.fft.ifft2(np.fft.fft2(image.variance) * np.fft.fft2(kernel ** 2))

    return image_variance_corr


def noise_kernels(science, reference):
    """
    Calculate the convolution kernels used in the noise correction
    
    The kernels calculated here are used in the convolution of the noise images
    that are used in the noise corrected matched filter images. They are 
    defined in equation 28 and 29 of Zackay, Ofek, Gal-Yam 2016.

    Parameters
    science : PyZOGY.ImageClass
        ImageClass instance created from the science image.
    reference : PyZOGY.ImageClass
        ImageClass instance created from the reference image.

    Returns
    -------
    science_kernel : numpy.ndarray
        Kernel for the convolution of arrays derived from the science image.
    reference_kernel : numpy.ndarray
        Kernel for the convolution of arrays derived from the reference image.
    """

    science_psf_fft = np.fft.fft2(science.psf)
    reference_psf_fft = np.fft.fft2(reference.psf)
    denominator = reference.background_std ** 2 * science.zero_point ** 2 * abs(science_psf_fft) ** 2
    denominator += science.background_std ** 2 * reference.zero_point ** 2 * abs(reference_psf_fft) ** 2

    science_kernel_fft = science.zero_point * reference.zero_point ** 2
    science_kernel_fft *= np.conj(reference_psf_fft) * abs(science_psf_fft) ** 2
    science_kernel_fft /= denominator
    science_kernel = np.fft.ifft2(science_kernel_fft)

    reference_kernel_fft = reference.zero_point * science.zero_point ** 2
    reference_kernel_fft *= np.conj(science_psf_fft) * abs(reference_psf_fft) ** 2
    reference_kernel_fft /= denominator
    reference_kernel = np.fft.ifft2(reference_kernel_fft)

    return science_kernel, reference_kernel


def registration_noise(image, kernel):
    """
    Calculate the registration noise for the noise correction
    
    Calculates the astrometric registration noise image. This noise image is
    used in the calculation of the noise corrected matched filter image.

    Parameters
    ----------
    image : PyZOGY.ImageClass
        ImageClass instance with registration_noise attribute defined.
    kernel : numpy.ndarray
        Convolution kernel for the noise image. This comes from the function
        called noise_kernels.
    
    Returns
    -------
    reg_variance : numpy.ndarray
        Noise image due to uncertainty in the image registration.
    """

    matched_part = np.fft.ifft2(np.fft.fft2(image) * np.fft.fft2(kernel))
    gradient = np.gradient(matched_part)
    # registration_noise is (x, y), gradient is (row, col)
    reg_variance = image.registration_noise[1] ** 2 * gradient[0] ** 2
    reg_variance += image.registration_noise[0] ** 2 * gradient[1] ** 2

    return reg_variance


def correct_matched_filter_image(science, reference):
    """
    Calculate the noise corrected matched filter image
    
    Computes the total noise used for the noise corrected matched filter image
    as defined in equation 25 of Zackay, Ofek, Gal-Yam 2016. This will work
    with the default read_noise and registration_noise, but it may not give
    a meaningful result.

    Parameters
    ----------
    science : PyZOGY.ImageClass
        ImageClass instance created from the science image.
    reference : PyZOGY.ImageClass
        ImageClass instance created from the reference image.

    Returns
    -------
    noise : numpy.ndarray
        The total noise in the matched filter image.
    """

    science_kernel, reference_kernel = noise_kernels(science, reference)
    science_source_noise = source_noise(science, science_kernel)
    reference_source_noise = source_noise(reference, reference_kernel)
    science_registration_noise = registration_noise(science, science_kernel)
    reference_registration_noise = registration_noise(reference, reference_kernel)
    noise = science_source_noise + reference_source_noise + science_registration_noise + reference_registration_noise
    return noise


def photometric_matched_filter_image(science, reference, matched_filter):
    """
    Calculate the photometry on the matched filter image
    """
    # note this may do exactly what another function above does
    # check this out later.

    science_psf_fft = np.fft.fft2(science.psf)
    reference_psf_fft = np.fft.fft2(reference.psf)
    zero_point = science.zero_point ** 2 * reference.zero_point ** 2
    zero_point *= abs(science_psf_fft) ** 2 * abs(reference_psf_fft) ** 2
    denominator = reference.background_std ** 2 * science.zero_point ** 2 * abs(science_psf_fft) ** 2
    denominator += science.background_std ** 2 * reference.zero_point ** 2 * abs(reference_psf_fft) ** 2
    zero_point /= denominator
    photometric_matched_filter = matched_filter / np.sum(zero_point)

    return photometric_matched_filter


def normalize_difference_image(difference, difference_image_zero_point, science, reference, normalization='reference'):
    """
    Normalize to user's choice of image
    
    Normalizes the difference image into the photometric system of the science
    image, reference image, or leave un-normalized.

    Parameters
    ----------
    difference : numpy.ndarray
        Difference image as calculated by calculate_difference_image.
    difference_image_zero_point : float
        Flux based zero point of the difference image above.
    science : PyZOGY.ImageClass
        ImageClass instance created from the science image.
    reference : PyZOGY.ImageClass
        ImageClass instance created from the reference image.
    normalization : str, optional
        Normalization choice. Options are 'reference', 'science', or 'none'.

    Returns
    -------
    difference_image : numpy.ndarray
        Normalized difference image.
    """

    if normalization == 'reference' or normalization == 't':
        difference_image = difference * reference.zero_point / difference_image_zero_point
    elif normalization == 'science' or normalization == 'i':
        difference_image = difference * science.zero_point / difference_image_zero_point
    else:
        difference_image = difference

    logging.info('Difference normalized to {}'.format(normalization))
    return difference_image


def run_subtraction(science_image, reference_image, science_psf, reference_psf, output='output.fits',
                    science_mask=None, reference_mask=None, n_stamps=1, normalization='reference',
                    science_saturation=np.inf, reference_saturation=np.inf, science_variance=None,
                    reference_variance=None, matched_filter=False, photometry=False,
                    gain_ratio=np.inf, gain_mask=None, use_pixels=False, sigma_cut=5., show=False, percent=99,
                    corrected=False, use_mask_for_gain=True, max_iterations=5, size_cut=False, pixstack_limit=None):
    """
    Run full subtraction given filenames and parameters
    
    Main function for users who don't want to use the ImageClass. This function
    lets the user put in all the arguments by hand and then creates the 
    ImageClass instances.

    Parameters
    ----------
    science_image : numpy.ndarray
        Science image to compare to reference image.
    reference_image : numpy.ndarray
        Reference image to subtract from science.
    science_psf : numpy.ndarray
        PSF of the science image.
    reference_psf : numpy.ndarray
        PSF of the reference image.
    output : str, optional, optional
        File name to save image to. Set to None to avoid writing output.
    science_mask : str, optional
        Name of the FITS file holding the science image mask.
    reference_mask : str, optional
        Name of the FITS file holding the reference image mask.
    n_stamps : int, optional
        Number of stamps to use while fitting background.
    normalization : str, optional
        Normalize difference image to 'reference', 'science', or 'none'.
    science_saturation : float, optional
        Maximum usable pixel value in science image.
    reference_saturation : float, optional
        Maximum usable pixel value in reference image.
    science_variance : numpy.ndarray or float, optional
        Variance of the science image
    reference_variance : numpy.ndarray or float, optional
        Variance of the reference image.
    matched_filter : bool, optional
        Calculate the matched filter image.
    photometry : bool, optional
        Photometrically normalize the matched filter image.
    gain_ratio : float, optional
        Ratio between the flux based zero points of the images.
    gain_mask : numpy.ndarray or str, optional
        Array or FITS image of pixels to use in gain matching.
    use_pixels : bool, optional
        Use pixels (True) or stars (False) to match gains.
    sigma_cut : float, optional
        Threshold (in standard deviations) to extract a star from the image (`thresh` in `sep.extract`).
    show : bool, optional
        Show debugging plots.
    percent : float, optional
        Percentile cutoff for gain matching.
    corrected : bool, optional
        Noise correct matched filter image.
    use_mask_for_gain : bool, optional
        Set to False in order to ignore the input masks when calculating the gain ratio.
    max_iterations : int, optional
        Maximum number of iterations to reconvolve the images for gain matching.
    size_cut : bool, optinal
        Ignores unusually large/small sources for gain matching (assumes most sources are real).
    pixstack_limit : int
        Number of active object pixels in Sep, set with sep.set_extract_pixstack
        
    Returns
    -------
    difference_image : numpy.ndarray
        The difference between science and reference images.
    """
    
    science = ImageClass(science_image, science_psf, science_mask, n_stamps, science_saturation, science_variance)
    reference = ImageClass(reference_image, reference_psf, reference_mask, n_stamps, reference_saturation,
                           reference_variance)
    difference = calculate_difference_image(science, reference, gain_ratio, gain_mask, sigma_cut, use_pixels, show,
                                            percent, use_mask_for_gain, max_iterations, size_cut, pixstack_limit)
    difference_zero_point = calculate_difference_image_zero_point(science, reference)
    difference_psf = calculate_difference_psf(science, reference, difference_zero_point)
    normalized_difference = normalize_difference_image(difference, difference_zero_point, science, reference,
                                                       normalization)

    if output is not None:
        save_difference_image_to_file(normalized_difference, science, normalization, output)
        save_difference_psf_to_file(difference_psf, output.replace('.fits', '.psf.fits'))

    if matched_filter:
        matched_filter_image = calculate_matched_filter_image(difference, difference_psf, difference_zero_point)
        if photometry and corrected:
            logging.error('Photometric matched filter and noise corrected matched filter are incompatible')
        if photometry:
            matched_filter_image = photometric_matched_filter_image(science, reference, matched_filter_image)
        elif corrected:
            matched_filter_image /= np.sqrt(correct_matched_filter_image(science, reference))
        fits.writeto(matched_filter, np.real(matched_filter_image), science.header, output_verify='warn', **overwrite)
        logging.info('Wrote matched filter image to {}'.format(matched_filter))
            
    return normalized_difference, difference_psf


def save_difference_image_to_file(difference_image, science, normalization, output):
    """
    Save difference image to file.
    
    Normalize and save difference image to file. This also copies over the 
    FITS header of the science image.

    Parameters
    ----------
    difference_image : numpy.ndarray
        Difference image
    science : PyZOGY.ImageClass
        ImageClass instance created from the science image.
    normalization : str
        Normalize to 'reference', 'science', or 'none'.
    output : str
        File to save FITS image to.
    """

    hdu = fits.PrimaryHDU(difference_image)
    hdu.header = science.header.copy()
    hdu.header['PHOTNORM'] = normalization
    hdu.writeto(output, output_verify='warn', **overwrite)
    logging.info('Wrote difference image to {}'.format(output))


def save_difference_psf_to_file(difference_psf, output):
    """
    Save difference image psf to file.
    
    Save the PSF of the difference image to a FITS file.

    Parameters
    ----------
    difference_psf : numpy.ndarray
        PSF of the difference image.
    output : str
        File to save FITS image to.
    """
    real_part = np.real(difference_psf)
    center = np.array(real_part.shape) / 2
    centered_psf = np.roll(real_part, center.astype(int), (0, 1))
    fits.writeto(output, centered_psf, output_verify='warn', **overwrite)
    logging.info('Wrote difference psf to {}'.format(output))
