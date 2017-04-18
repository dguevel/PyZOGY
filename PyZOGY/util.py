import numpy as np
import scipy
import statsmodels.api as stats


def make_pixel_mask(image, saturation, input_mask=None):
    """Make a pixel mask that marks saturated pixels; optionally join with input_mask"""

    if input_mask is not None:
        new_mask = input_mask
    else:
        new_mask = np.zeros(image.shape)

    new_mask[image >= saturation] = 1
    return new_mask


def center_psf(psf):
    """Center psf at (0,0) based on max value"""

    peak = np.unravel_index(psf.argmax(), psf.shape)

    psf = np.roll(psf, -peak[0], 0)
    psf = np.roll(psf, -peak[1], 1)
    return psf


def fit_noise(data, n_stamps=1, mode='iqr'):
    """Find the standard deviation of the image background; returns standard deviation, median"""

    median_small = np.zeros([n_stamps, n_stamps])
    std_small = np.zeros([n_stamps, n_stamps])
    for y_stamp in range(n_stamps):
        for x_stamp in range(n_stamps):
            y_index = [y_stamp * data.shape[0] // n_stamps, (y_stamp + 1) * data.shape[0] // n_stamps]
            x_index = [x_stamp * data.shape[1] // n_stamps, (x_stamp + 1) * data.shape[1] // n_stamps]
            stamp_data = data[y_index[0]: y_index[1], x_index[0]: x_index[1]]
            if mode == 'gaussian':
                trimmed_stamp_data = stamp_data[stamp_data < np.percentile(stamp_data, 90)]
                trimmed_stamp_data = trimmed_stamp_data[trimmed_stamp_data != 0]
                histogram_data = np.histogram(trimmed_stamp_data, bins=100)
                x = histogram_data[1][:-1]
                y = histogram_data[0]
                guess = [np.max(y), np.median(trimmed_stamp_data), np.std(trimmed_stamp_data)]
                parameters, covariance = scipy.optimize.curve_fit(gauss, x, y, p0=guess, maxfev=1600)
                median_small[y_stamp, x_stamp] = parameters[1]
                std_small[y_stamp, x_stamp] = parameters[2]
            elif mode == 'iqr':
                quartile25, median, quartile75 = np.percentile(data, (25, 50, 75))
                median_small[y_stamp, x_stamp] = median
                # 0.741301109 is a tuning parameter that scales iqr to std
                std_small[y_stamp, x_stamp] = 0.741301109 * (quartile75 - quartile25)

    median = scipy.ndimage.zoom(median_small, [data.shape[0] / float(n_stamps), data.shape[1] / float(n_stamps)])
    std = scipy.ndimage.zoom(std_small, [data.shape[0] / float(n_stamps), data.shape[1] / float(n_stamps)])

    return std, median


def gauss(position, amplitude, median, std):
    """Return a gaussian function"""

    return amplitude * np.exp(-(position - median) ** 2 / (2 * std ** 2))


def interpolate_bad_pixels(image, mask, median_size=6):
    """Interpolate over bad pixels using a global median"""

    # from http://stackoverflow.com/questions/18951500/automatically-remove-hot-dead-pixels-from-an-image-in-python
    pix = np.transpose(np.where(mask == 1))
    blurred = scipy.ndimage.median_filter(image, size=median_size)
    interpolated_image = np.copy(image)
    for y, x in pix:
        interpolated_image[y, x] = blurred[y, x]

    return interpolated_image


def resize_psf(psf, shape):
    """Resize centered (0,0) psf to larger shape"""

    psf_extended = np.pad(psf, ((0, shape[0] - psf.shape[0]), (0, shape[1] - psf.shape[1])),
                          mode='constant', constant_values=0.)
    return psf_extended


def pad_to_power2(data):
    """Pad arrays to the nearest power of two"""

    n = 0
    defecit = [0, 0]
    while (data.shape[0] > (2 ** n)) or (data.shape[1] > (2 ** n)):
        n += 1
        defecit = [(2 ** n) - data.shape[0], (2 ** n) - data.shape[1]]
    padded_data = np.pad(data, ((0, defecit[0]), (0, defecit[1])), mode='constant', constant_values=np.median(data))
    return padded_data


def solve_iteratively(science, reference,
                      mask_tolerance=10e-5, gain_tolerance=10e-6, max_iterations=5, sigma_cut=1):
    """Solve for linear fit iteratively"""

    gain = 1.
    gain0 = 10e5
    i = 0
    # pad image to power of two to speed fft
    old_size = science.image_data.shape
    science_image = pad_to_power2(science.image_data)
    reference_image = pad_to_power2(reference.image_data)

    science_psf = center_psf(resize_psf(science.raw_psf_data, science_image.shape))
    science_psf /= np.sum(science.raw_psf_data)
    reference_psf = center_psf(resize_psf(reference.raw_psf_data, reference_image.shape))
    reference_psf /= np.sum(reference.raw_psf_data)

    science_std = pad_to_power2(science.background_std)
    reference_std = pad_to_power2(reference.background_std)

    science_mask = pad_to_power2(science.pixel_mask)
    reference_mask = pad_to_power2(reference.pixel_mask)
    # fft arrays
    science_image_fft = np.fft.fft2(science_image)
    reference_image_fft = np.fft.fft2(reference_image)
    science_psf_fft = np.fft.fft2(science_psf)
    reference_psf_fft = np.fft.fft2(reference_psf)

    while abs(gain - gain0) > gain_tolerance:

        # calculate the psf in the difference image to convolve masks
        # not a simple convolution of the two PSF's; see the paper for details
        difference_zero_point = gain / np.sqrt(science_std ** 2 + reference_std ** 2 * gain ** 2)
        denominator = science_std ** 2 * abs(reference_psf_fft) ** 2
        denominator += reference_std ** 2 * gain ** 2 * abs(science_psf_fft) ** 2
        difference_psf_fft = gain * science_psf_fft * reference_psf_fft / (difference_zero_point * np.sqrt(denominator))

        # convolve masks with difference psf to mask all pixels within a psf radius
        # this is important to prevent convolutions of saturated pixels from affecting the fit
        science_mask_convolved = np.fft.ifft2(difference_psf_fft * np.fft.fft2(science_mask))
        science_mask_convolved[science_mask_convolved > mask_tolerance] = 1
        science_mask_convolved = np.real(science_mask_convolved).astype(int)
        reference_mask_convolved = np.fft.ifft2(difference_psf_fft * np.fft.fft2(reference_mask))
        reference_mask_convolved[reference_mask_convolved > mask_tolerance] = 1
        reference_mask_convolved = np.real(reference_mask_convolved).astype(int)

        # do the convolutions on the images
        denominator = science_std ** 2 * abs(reference_psf_fft) ** 2
        denominator += gain ** 2 * reference_std ** 2 * abs(science_psf_fft) ** 2

        science_convolved_image_fft = reference_psf_fft * science_image_fft / np.sqrt(denominator)
        reference_convolved_image_fft = science_psf_fft * reference_image_fft / np.sqrt(denominator)

        science_convolved_image = np.real(np.fft.ifft2(science_convolved_image_fft))
        reference_convolved_image = np.real(np.fft.ifft2(reference_convolved_image_fft))

        # remove pixels less than sigma_cut above sky level to speed fitting
        science_min = np.median(science_convolved_image) + sigma_cut * np.std(science_convolved_image)
        science_convolved_image[science_convolved_image < science_min] = np.nan

        reference_min = np.median(reference_convolved_image) + sigma_cut * np.std(reference_convolved_image)
        reference_convolved_image[reference_convolved_image < reference_min] = np.nan

        # remove pixels marked in the convolved mask
        science_convolved_image[science_mask_convolved == 1] = np.nan
        reference_convolved_image[reference_mask_convolved == 1] = np.nan

        # remove power of 2 padding
        science_convolved_image = science_convolved_image[: old_size[0], : old_size[1]]
        reference_convolved_image = reference_convolved_image[: old_size[0], : old_size[1]]

        # join the two criteria for pixel inclusion
        science_good_pix = ~np.isnan(science_convolved_image)
        reference_good_pix = ~np.isnan(reference_convolved_image)
        good_pix_in_common = np.logical_and(science_good_pix, reference_good_pix)

        # flatten into 1d arrays of good pixels
        science_convolved_image_flatten = science_convolved_image[good_pix_in_common]
        reference_convolved_image_flatten = reference_convolved_image[good_pix_in_common]

        # do a linear robust regression between convolved images
        gain0 = gain
        x = reference_convolved_image_flatten
        y = science_convolved_image_flatten
        robust_fit = stats.RLM(y, x).fit()
        parameters = robust_fit.params
        gain = parameters[0]

        if i == max_iterations:
            break
        i += 1
        print('Iteration {0}:'.format(i))
        print('Gain = {0}'.format(gain))

    print('Fit done in {} iterations'.format(i))
    variance = robust_fit.bcov_scaled[0, 0]
    print('Gain = ' + str(gain))
    print('Gain Variance = ' + str(variance))

    return gain
