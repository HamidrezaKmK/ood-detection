"""
This package contains all the OOD methods that rely on the fact that you can
linearly approximate your generative model for a specific point in a small neighborhood. 
"""

from .linear_methods import LatentScore, RadiiTrend

from .latent_statistics import ParallelogramLogCDF, EllipsoidCDFStatsCalculator, GaussianConvolutionStatsCalculator

from .encoding_model import EncodingFlow, EncodingVAE