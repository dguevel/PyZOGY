import numpy as np
import scipy
import statsmodels.api as stats


def center_psf(psf):
    """Center psf at (0,0) based on max value"""

    peak = np.unravel_index(psf.argmax(), psf.shape)

    psf = np.roll(psf, -peak[0], 0)
    psf = np.roll(psf, -peak[1], 1)
    return psf


def fit_noise(data, n_stamps=1):
    """Find the standard deviation of the image background; returns standard deviation, median"""

    median_small = np.zeros([n_stamps, n_stamps])
    std_small = np.zeros([n_stamps, n_stamps])
    for y_stamp in range(n_stamps):
        for x_stamp in range(n_stamps):
            y_index = [y_stamp * data.shape[0] / n_stamps, (y_stamp + 1) * data.shape[0] / n_stamps]
            x_index = [x_stamp * data.shape[1] / n_stamps, (x_stamp + 1) * data.shape[1] / n_stamps]
            stamp_data = data[y_index[0]: y_index[1], x_index[0]: x_index[1]]
            trimmed_stamp_data = stamp_data[stamp_data < np.percentile(stamp_data, 90)]
            trimmed_stamp_data = trimmed_stamp_data[trimmed_stamp_data != 0]
            histogram_data = np.histogram(trimmed_stamp_data, bins=100)
            x = histogram_data[1][:-1]
            y = histogram_data[0]
            guess = [np.max(y), np.median(trimmed_stamp_data), np.std(trimmed_stamp_data)]
            parameters, covariance = scipy.optimize.curve_fit(gauss, x, y, p0=guess, maxfev=1600)
            median_small[y_stamp, x_stamp] = parameters[1]
            std_small[y_stamp, x_stamp] = parameters[2]

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
    padded_data = np.pad(data, ((0, defecit[0]), (0, defecit[1])), mode='constant', constant_values=0.)
    return padded_data


def solve_iteratively(science, reference):
    """Solve for linear fit iteratively"""

    gain_tolerance = 0.001
    gain = 1.
    gain0 = 10e5
    i = 0
    max_iterations = 5

    # trim image to speed fitting
    old_size = science.image_data.shape
    science_image = pad_to_power2(science.image_data)
    reference_image = pad_to_power2(reference.image_data)
    science_psf = resize_psf(center_psf(science.raw_psf_data), science_image.shape)
    reference_psf = resize_psf(center_psf(reference.raw_psf_data), reference_image.shape)
    science_std = pad_to_power2(science.background_std)
    reference_std = pad_to_power2(reference.background_std)

    science_image_fft = np.fft.fft2(science_image)
    reference_image_fft = np.fft.fft2(reference_image)
    science_psf_fft = np.fft.fft2(science_psf)
    reference_psf_fft = np.fft.fft2(reference_psf)
    while abs(gain - gain0) > gain_tolerance:

        denominator = science_std ** 2 * abs(reference_psf_fft) ** 2
        denominator += gain ** 2 * reference_std ** 2 * abs(science_psf_fft) ** 2
        science_convolved_image_fft = reference_psf_fft * science_image_fft / np.sqrt(denominator)
        reference_convolved_image_fft = science_psf_fft * reference_image_fft / np.sqrt(denominator)
        science_convolved_image = np.real(np.fft.ifft2(science_convolved_image_fft))[0: old_size[0], 0: old_size[1]]
        reference_convolved_image = np.real(np.fft.ifft2(reference_convolved_image_fft))[0: old_size[0], 0: old_size[1]]
        science_convolved_image_flatten = science_convolved_image.flatten()
        reference_convolved_image_flatten = reference_convolved_image.flatten()

        # remove pixels less than one sigma above sky level to speed fitting
        science_good_pix = np.where([science_convolved_image_flatten > np.median(science_convolved_image_flatten)
                                     + np.std(science_convolved_image_flatten)])
        reference_good_pix = np.where([reference_convolved_image_flatten > np.median(reference_convolved_image_flatten)
                                       + np.std(reference_convolved_image_flatten)])

        good_pix_in_common = np.intersect1d(science_good_pix, reference_good_pix)
        science_convolved_image_flatten = science_convolved_image_flatten[good_pix_in_common]
        reference_convolved_image_flatten = reference_convolved_image_flatten[good_pix_in_common]

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

    covariance = robust_fit.bcov_scaled[0, 0]
    print('Gain = ' + str(gain))
    print('Gain Variance = ' + str(covariance))

    return gain
