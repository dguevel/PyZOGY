import numpy

class BaseImage(numpy.ma.MaskedArray):
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

        obj = numpy.asarray(data, **kwargs).view(cls)
        obj._fft = None
        obj._fft_pad = None

        return obj


    def __array_finalize__(self, obj):
        if obj is None:
            return
        obj._fft = getattr(obj, '_fft', None)
        obj._fft_pad= getattr(obj, '_pad', None)


    def fft(self):
        """
        Calculate the object's padded FFT.

        Use numpy to pad and calculate the FFT. This function stores the FFT
        in memory to avoid repeated calculation. If you don't want this, use
        numpy.fft directly on the instance.

        Returns
        -------
        fft : numpy.ma.MaskedArray
            Fourier transformed masked array.
        """

        if self._fft is None:
            self._fft = numpy.ma.MaskedArray(numpy.fft.fft2(self.pad()), mask=self.mask)
        return self._fft


    def fft_pad(self):
        """
        Pad an array to a power of two.
        
        Use numpy to pad the edges to the next highest power of two to 
        speed up FFT and to avoid effects of circular convolution.

        Returns
        -------
        fft_pad : numpy.ma.MaskedArray
            Padded masked array.
        """

        if self._pad is None:
            constant = np.median(self)
            n = 0
            defecit = [0, 0]
            while (data.shape[0] > (2 ** n)) or (data.shape[1] > (2 ** n)):
                n += 1
                defecit = [(2 ** n) - data.shape[0], (2 ** n) - data.shape[1]]
            padding = ((0, defecit[0]), (0, defecit[1])),
            padded_data = numpy.pad(self, padding, mode='constant', constant_values=constant)
            padded_mask = numpy.pad(self.mask, padding, mode='constant', constant_values=True)
            self._fft_pad = numpy.ma.MaskedArray(padded_mask, mask=padded_mask)
        return self._fft_pad


class PSF(BaseImage):
    def resize(self, shape):
        """
        Resize centered psf to larger shape.

        Use numpy to resize psf to a larger shape for convolution with image.
        
        Parameters
        ----------
        shape : array-like
            Desired PSF dimensions.

        Returns
        -------
        extended : numpy.ndarray
            Padded PSF.
        """

        if self._extended is None:
            shape_diff = [2 * [(shape[0] - psf.shape[0]) // 2], 2 * [(shape[1] - psf.shape[1]) // 2]]
            if (shape[0] - psf.shape[0]) % 2 != 0:
                shape_diff[0][1] += 1
            if (shape[1] - psf.shape[1]) % 2 != 0:
                shape_diff[1][1] += 1

            padding = ((shape_diff[0][0], shape_diff[0][1]), shape_diff[1][0], shape_diff[1][1]))

            self._extended = np.pad(psf, padding, mode='constant', constant_values=0.)
        return psf_extended

class Image(BaseImage):
    pass

class DifferenceImage(BaseImage):
    pass

class ScoreImage(BaseImage):
    pass

