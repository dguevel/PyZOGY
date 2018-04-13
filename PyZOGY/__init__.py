import numpy as np

class BaseImage(np.ndarray):
    """
    Base class for difference imaging.

    Base class includes members relevant to all derived classes like file
    IO and common array manipulations. The class will store derived arrays
    in memory to avoid repetitive calculation.

    Parameters
    ----------
    array : array-like
        Input array data.

    Methods
    -------
    fft()
        Calculate the FFT of an array.
    write(filename)
        Write to a FITS file.
    """

    def __new__(cls, data, **kwargs):

        obj = np.asarray(data, **kwargs).view(cls)
        obj._fft = None
        obj._pad = None

        return obj


    def __array_finalize__(self, obj):
        if obj is None:
            return
        self._fft = getattr(obj, '_fft', None)
        self._pad= getattr(obj, '_pad', None)


    def fft(self):
        """
        Calculate the object's padded FFT.

        Use np to pad and calculate the FFT. This function stores the FFT
        in memory to avoid repeated calculation. If you don't want this, use
        np.fft directly on the instance.

        Returns
        -------
        fft : np.ma.MaskedArray
            Fourier transformed masked array.
        """

        if self._fft is None:
            self._fft = np.fft.fft2(self.pad()).real
        return self._fft


    def pad(self):
        """
        Pad an array to a power of two.
        
        Use np to pad the edges to the next highest power of two to 
        speed up FFT and to avoid effects of circular convolution.

        Returns
        -------
        fft_pad : np.ma.MaskedArray
            Padded masked array.
        """

        if self._pad is None:
            constant = np.median(self)
            n = 0
            defecit = [0, 0]
            while (self.shape[0] > (2 ** n)) or (self.shape[1] > (2 ** n)):
                n += 1
                defecit = [(2 ** n) - self.shape[0], (2 ** n) - self.shape[1]]
            padding = ((0, defecit[0]), (0, defecit[1]))
            padded_data = np.pad(self, padding, mode='constant', constant_values=constant)
            self._pad = padded_data
        return self._pad


class PSF(BaseImage):
    def __new__(cls, data, **kwargs):

        obj = BaseImage(data, **kwargs).view(cls)
        obj._fft = None
        obj._fft_pad = None

        return obj


    def __array_finalize__(self, obj):
        if obj is None:
            return

    def resize(self, shape):
        """
        Resize centered psf to larger shape.

        Use np to resize psf to a larger shape for convolution with image.
        
        Parameters
        ----------
        shape : array-like
            Desired PSF dimensions.

        Returns
        -------
        extended : np.ndarray
            Padded PSF.
        """

        if self._extended is None:
            shape_diff = [2 * [(shape[0] - psf.shape[0]) // 2], 2 * [(shape[1] - psf.shape[1]) // 2]]
            if (shape[0] - psf.shape[0]) % 2 != 0:
                shape_diff[0][1] += 1
            if (shape[1] - psf.shape[1]) % 2 != 0:
                shape_diff[1][1] += 1

            padding = ((shape_diff[0][0], shape_diff[0][1]), (shape_diff[1][0], shape_diff[1][1]))

            self._extended = np.pad(psf, padding, mode='constant', constant_values=0.)
        return psf_extended

    def mask_saturated_pix(self):
        """Make a pixel mask that marks saturated pixels; optionally join with input_mask"""

        input_mask = np.zeros(image.shape)

        input_mask[image >= self.saturation] = 1
        input_mask = input_mask.astype(bool)


def pad_to_power2(data, value='median'):

class Image(BaseImage):
    def __new__(cls, data, psf, mask=None, **kwargs):

        obj = BaseImage(data, **kwargs).view(cls)
        obj.zero_point = np.inf
        obj.psf = psf
        obj.background_std = np.inf
        
        if mask = None:
            obj.mask = np.zeros(obj.shape)
        
        return obj


    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.zero_point = getattr(obj, 'zero_point', np.inf)
        self.psf = getattr(obj, 'psf', None)
        self.background_std= getattr(obj, 'background_std', None)



    def difference_image(self, reference, gain_mask=None, use_pixels=False, show=False, percent=99):
        """
        Calculate the difference image using the Zackay algorithm.
        
        This is the main function that calculates the difference image using the 
        Zackey, Ofek, Gal-Yam 2016. It operates on ImageClass objects defined in
        image_class.py. The function will fit the gain ratio if not provided. 
        Ultimately this calculates equation 13 in Zackey, Ofek, Gal-Yam 2016.
        
        Parameters
        ----------
        reference : PyZOGY.Image
            ImageClass instance created from the reference image.
        gain_mask : str or numpy.ndarray, optional
            Array or FITS file holding an array of pixels to use when fitting
            the gain ratio.
        use_pixels : bool, optional
            Fit the gain ratio using pixels (True) or stars (False) in image.
        show : bool, optional
            Display debuggin plots during fitting.
        percent : float, optional
            Percentile cutoff to use for fitting the gain ratio. 

        Returns
        -------
        difference_image : numpy.ndarray
            The difference between science and reference images.
        """

        # match the gains, set reference to their one, self to 1.
        if self.zero_point == np.inf or reference.zero_point == np.inf:
            if gain_mask is not None:
                self.mask[gain_mask == 1] = 1
                reference.mask[gain_mask == 1] = 1
            self.zero_point = util.solve_iteratively(science, reference, use_pixels=use_pixels, show=show, percent=percent)
            reference.zero_point = 1.

        # calculate difference image
        denominator = self.background_std ** 2 * reference.zero_point ** 2 * abs(reference.psf.fft()) ** 2
        denominator += reference.background_std ** 2 * self.zero_point ** 2 * abs(self.psf.fft()) ** 2
        difference_image_fft = self.fft() * reference.psf.fft() * reference.zero_point
        difference_image_fft -= reference.image.fft() * self.psf.fft() * self.zero_point
        difference_image_fft /= np.sqrt(denominator)
        difference_image = np.fft.ifft2(difference_image_fft)

        difference = DifferenceImage(difference_image, self, reference)

        return difference

class DifferenceImage(BaseImage):
    def zero_point(self):
        """
        Calculate the flux based zero point of the difference image.
        
        Calculate the difference image flux based zero point using equation 15 of 
        Zackey, Ofek, Gal-Yam 2016.

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

        denominator = self.science.background_std ** 2 * self.reference.zero_point ** 2
        denominator += self.reference.background_std ** 2 * self.science.zero_point ** 2
        zero_point = self.science.zero_point * self.reference.zero_point / np.sqrt(denominator)

        logging.info('Global difference image zero point is {}'.format(np.mean(difference_image_zero_point)))
        return zero_point


    def psf(self):
        """
        Calculate the PSF of the difference image.
        
        Calculactes the PSF of the difference image using equation 17 of Zackey,
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

        denominator = self.science.background_std ** 2 * self.reference.zero_point ** 2 * abs(self.reference.psf.fft()) ** 2
        denominator += reference.background_std ** 2 * science.zero_point ** 2 * abs(self.science.psf.fft) ** 2

        psf_fft = science.zero_point * science_psf_fft * reference_psf_fft
        psf_fft /= self.zero_point * np.sqrt(denominator)
        psf = PSF(np.fft.ifft2(psf_fft))

        return psf

    pass

class ScoreImage(BaseImage):
    pass

