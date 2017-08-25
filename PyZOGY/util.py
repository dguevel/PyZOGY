import numpy as np
import scipy.ndimage
import statsmodels.api as stats
from astropy.io import fits
import matplotlib.pyplot as plt
import sep


def make_mask(image, saturation, input_mask=''):
    """Make a pixel mask that marks saturated pixels; optionally join with input_mask"""

    if input_mask != '':
        new_mask = fits.getdata(input_mask)
    else:
        new_mask = np.zeros(image.shape)

    new_mask[image >= saturation] = 1
    return new_mask


def center_psf(psf):
    """Center psf at (0,0) based on max value"""

    peak = np.array(np.unravel_index(psf.argmax(), psf.shape))
    psf = np.roll(psf, -peak, (0, 1))
    return psf


def fit_noise(data, n_stamps=1, mode='sep', output_name='background'):
    """Find the standard deviation of the image background; returns standard deviation, median"""

    median_small = np.zeros([n_stamps, n_stamps])
    std_small = np.zeros([n_stamps, n_stamps])
    if mode == 'sep':
        background = sep.Background(np.ascontiguousarray(data.data).byteswap().newbyteorder())
        median = background.back()
        std = background.rms()
    else:
        for y_stamp in range(n_stamps):
            for x_stamp in range(n_stamps):
                y_index = [y_stamp * data.shape[0] // n_stamps, (y_stamp + 1) * data.shape[0] // n_stamps]
                x_index = [x_stamp * data.shape[1] // n_stamps, (x_stamp + 1) * data.shape[1] // n_stamps]
                stamp_data = data[y_index[0]: y_index[1], x_index[0]: x_index[1]].compressed()
                if mode == 'iqr':
                    quartile25, median, quartile75 = np.percentile(stamp_data, (25, 50, 75))
                    median_small[y_stamp, x_stamp] = median
                    # 0.741301109 is a parameter that scales iqr to std
                    std_small[y_stamp, x_stamp] = 0.741301109 * (quartile75 - quartile25)
                elif mode == 'mad':
                    median = np.median(stamp_data)
                    absdev = np.abs(stamp_data - median)
                    mad = np.median(absdev)
                    median_small[y_stamp, x_stamp] = median
                    std_small[y_stamp, x_stamp] = 1.4826 * mad

        median = scipy.ndimage.zoom(median_small, [data.shape[0] / float(n_stamps), data.shape[1] / float(n_stamps)])
        std = scipy.ndimage.zoom(std_small, [data.shape[0] / float(n_stamps), data.shape[1] / float(n_stamps)])

    return std, median


def interpolate_bad_pixels(image, median_size=6):
    """Interpolate over bad pixels using a global median; needs a mask"""

    # from http://stackoverflow.com/questions/18951500/automatically-remove-hot-dead-pixels-from-an-image-in-python
    pix = np.transpose(np.where(image.mask))
    blurred = scipy.ndimage.median_filter(image, size=median_size)
    interpolated_image = np.copy(image)
    for y, x in pix:
        interpolated_image[y, x] = blurred[y, x]

    return interpolated_image


def join_images(science_raw, science_mask, reference_raw, reference_mask, sigma_cut, use_pixels=False, show=False):
    """Join two images to fittable vectors"""

    science = np.ma.array(science_raw, mask=science_mask, copy=True)
    reference = np.ma.array(reference_raw, mask=reference_mask, copy=True)
    science_std = np.ma.std(science)
    reference_std = np.ma.std(reference)
    if use_pixels:
        # remove pixels less than sigma_cut above sky level to speed fitting
        science.mask[science < sigma_cut * science_std] = True
        reference.mask[reference_raw < sigma_cut * reference_std] = True

        # flatten into 1d arrays of good pixels
        science.mask |= reference.mask
        reference.mask |= science.mask
        science_flatten = science.compressed()
        reference_flatten = reference.compressed()
    else:
        science_sources = sep.extract(np.ascontiguousarray(science.data), thresh=sigma_cut, err=science_std, mask=np.ascontiguousarray(science.mask))
        reference_sources = sep.extract(np.ascontiguousarray(reference.data), thresh=sigma_cut, err=reference_std, mask=np.ascontiguousarray(reference.mask))
        dx = science_sources['x'] - np.atleast_2d(reference_sources['x']).T
        dy = science_sources['y'] - np.atleast_2d(reference_sources['y']).T
        sep2 = dx**2 + dy**2
        matches = np.min(sep2, axis=1) < 1.
        inds = np.argmin(sep2, axis=1)
        science_flatten = science_sources['flux'][inds][matches]
        reference_flatten = reference_sources['flux'][matches]

    if show:
        plt.ion()
        plt.figure(1)
        plt.clf()
        vmin, vmax = np.percentile(science, (1, 99))
        plt.imshow(science, vmin=vmin, vmax=vmax)
        plt.title('Science')
        if not use_pixels:
            plt.plot(reference_sources['x'][matches], reference_sources['y'][matches], 'o', mfc='none', mec='r')
        
        plt.figure(2)
        plt.clf()
        vmin, vmax = np.percentile(reference, (1, 99))
        plt.imshow(reference, vmin=vmin, vmax=vmax)
        plt.title('Reference')
        if not use_pixels:
            plt.plot(reference_sources['x'][matches], reference_sources['y'][matches], 'o', mfc='none', mec='r')
        
        plt.figure(3)
        plt.clf()
        plt.plot(reference_flatten, science_flatten, '.')
        plt.xlabel('Reference')
        plt.ylabel('Science')

    return reference_flatten, science_flatten


def resize_psf(psf, shape):
    """Resize centered (0,0) psf to larger shape"""

    psf_extended = np.pad(psf, ((0, shape[0] - psf.shape[0]), (0, shape[1] - psf.shape[1])),
                          mode='constant', constant_values=0.)
    return psf_extended


def pad_to_power2(data, value='median'):
    """Pad arrays to the nearest power of two"""

    if value == 'median':
        constant = np.median(data)
    else:
        constant = 0.
    n = 0
    defecit = [0, 0]
    while (data.shape[0] > (2 ** n)) or (data.shape[1] > (2 ** n)):
        n += 1
        defecit = [(2 ** n) - data.shape[0], (2 ** n) - data.shape[1]]
    padded_data = np.pad(data, ((0, defecit[0]), (0, defecit[1])), mode='constant', constant_values=constant)
    return padded_data


def solve_iteratively(science, reference, mask_tolerance=10e-5, gain_tolerance=10e-6,
                      max_iterations=5, sigma_cut=5, min_elements=800, use_pixels=False, show=False):
    """Solve for linear fit iteratively"""

    gain = 1.
    gain0 = 10e5
    i = 0
    # pad image to power of two to speed fft
    old_size = science.shape
    science_image = pad_to_power2(science)
    reference_image = pad_to_power2(reference)

    science_psf = center_psf(resize_psf(science.raw_psf, science_image.shape))
    science_psf /= np.sum(science.raw_psf)
    reference_psf = center_psf(resize_psf(reference.raw_psf, reference_image.shape))
    reference_psf /= np.sum(reference.raw_psf)

    science_std = pad_to_power2(science.background_std)
    reference_std = pad_to_power2(reference.background_std)

    science_mask = pad_to_power2(science.mask)
    reference_mask = pad_to_power2(reference.mask)

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

        # remove power of 2 padding
        science_convolved_image = science_convolved_image[: old_size[0], : old_size[1]]
        reference_convolved_image = reference_convolved_image[: old_size[0], : old_size[1]]
        science_mask_convolved = science_mask_convolved[: old_size[0], : old_size[1]]
        reference_mask_convolved = reference_mask_convolved[: old_size[0], : old_size[1]]

        # do a linear robust regression between convolved image
        gain0 = gain
        sigma = sigma_cut

        x, y = join_images(science_convolved_image, science_mask_convolved, reference_convolved_image, 
                           reference_mask_convolved, sigma, use_pixels, show)
        while (x.size < min_elements) or (y.size < min_elements):
            sigma -= 0.5
            x, y = join_images(science_convolved_image, science_mask_convolved, reference_convolved_image, 
                               reference_mask_convolved, sigma, use_pixels, show)
            if sigma == 0.:
                break
        robust_fit = stats.RLM(y, x).fit()
        parameters = robust_fit.params
        gain = parameters[0]
        if show:
            xfit = np.arange(np.max(x))
            plt.plot(xfit, gain*xfit)
            plt.legend()
            input('Press enter to continue to next iteration')

        if i == max_iterations:
            break
        i += 1
        print('Iteration {0}:'.format(i))
        print('Gain = {0}'.format(gain))

    print('Fit done in {} iterations'.format(i))

    return gain
