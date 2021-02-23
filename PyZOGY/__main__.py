#!/usr/bin/env python
import argparse
import PyZOGY.subtract
import numpy as np
import logging


def main():
    parser = argparse.ArgumentParser(description="Subtract images using ZOGY algorithm.")

    parser.add_argument('--science-image', dest='science_image', help='Science image to subtract')
    parser.add_argument('--science-psf', dest='science_psf', help='PSF for the science image')
    parser.add_argument('--output', dest='output', default='output.fits', help='Output file name')
    parser.add_argument('--science-mask', dest='science_mask', help='Mask for the science image', default=None)
    parser.add_argument('--reference-image', dest='reference_image', help='Reference image to subtract')
    parser.add_argument('--reference-psf', dest='reference_psf', help='PSF for the reference image')
    parser.add_argument('--reference-mask', dest='reference_mask', help='Mask for the reference image', default=None)

    parser.add_argument('--science-saturation', dest='science_saturation', help='Science image saturation value',
                        default=np.inf, type=float)
    parser.add_argument('--reference-saturation', dest='reference_saturation', help='Reference image saturation value',
                        default=np.inf, type=float)
    parser.add_argument('--science-variance', dest='science_variance', help='Science variance image',
                        default=None)
    parser.add_argument('--reference-variance', dest='reference_variance', help='Reference variance image',
                        default=None)

    parser.add_argument('--n-stamps', dest='n_stamps', help='Number of stamps to use when fitting the sky level',
                        default=1, type=int)
    parser.add_argument('--normalization', dest='normalization', help='Which image to normalize the difference to',
                        default='reference')
    parser.add_argument('--gain-ratio', help='Ratio of the science zero point to the reference zero point',
                        default=np.inf, type=float)
    parser.add_argument('--gain-mask', help='Additional mask for pixels not to be used in gain matching')
    parser.add_argument('--use-pixels', action='store_true', help='Use pixels for gain matching instead of stars')
    parser.add_argument('--show', action='store_true', help='Show plots during for gain matching')
    parser.add_argument('--sigma-cut', help='Threshold (in standard deviations) to extract a star from the image',
                        default=5., type=float)
    parser.add_argument('--no-mask-for-gain', help='Ignore the input masks when calculating the gain ratio',
                        action='store_false', dest='use_mask_for_gain')
    parser.add_argument('--max-iterations', default=5, type=int,
                        help='Maximum number of iterations to reconvolve the images for gain matching')
    parser.add_argument('--size-cut', action='store_true',
                        help='Ignore unusually large/small sources for gain matching (assumes most sources are real)')

    parser.add_argument('--matched-filter', help='Output filename for matched filter image')
    parser.add_argument('--correct', action='store_true', help='Correct matched filter image with noise')
    parser.add_argument('--photometry', action='store_true', help='Correct matched filter image to do photometry')

    parser.add_argument('--log', dest='log', help='Log output file', default='pyzogy.log')
    parser.add_argument('--percent', dest='percent', help='Pixel percentile for gain matching on pixels', default=99)
    parser.add_argument('--pixstack-limit', type=int, help='Modify set_extract_pixstack in Sep')

    args = parser.parse_args()

    logging.basicConfig(filename=args.log, level=logging.DEBUG, filemode='w')

    PyZOGY.subtract.run_subtraction(args.science_image,
                                    args.reference_image,
                                    args.science_psf,
                                    args.reference_psf,
                                    output=args.output,
                                    science_mask=args.science_mask,
                                    reference_mask=args.reference_mask,
                                    n_stamps=args.n_stamps,
                                    normalization=args.normalization,
                                    science_saturation=args.science_saturation,
                                    reference_saturation=args.reference_saturation,
                                    science_variance=args.science_variance,
                                    reference_variance=args.reference_variance,
                                    gain_ratio=args.gain_ratio,
                                    gain_mask=args.gain_mask,
                                    use_pixels=args.use_pixels,
                                    show=args.show,
                                    matched_filter=args.matched_filter,
                                    percent=args.percent,
                                    corrected=args.correct,
                                    photometry=args.photometry,
                                    sigma_cut=args.sigma_cut,
                                    use_mask_for_gain=args.use_mask_for_gain,
                                    max_iterations=args.max_iterations,
                                    size_cut=args.size_cut,
                                    pixstack_limit=args.pixstack_limit)

if __name__ == '__main__':
    main()
