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
        return obj


    def __array_finalize__(self, obj):
        if obj is None:
            return
        obj._fft = getattr(obj, '_fft', None)


    def fft(self):
        """
        Calculate the object's padded FFT.

        Use numpy to pad and calculate the FFT. This function stores the FFT
        in memory to avoid repeated calculation. If you don't want this, use
        numpy.fft directly on the instance.
        """

        if self._fft == None:
            self._fft = numpy.fft.fft2(self)
        return self._fft


    def fft_pad(self):
        """
        Pad an array to a power of two.
        
        Use numpy to pad the edges to the next highest power of two to 
        speed up.
        """

        pass


