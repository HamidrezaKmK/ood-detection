"""
This package contains all the OOD methods that rely on the fact that you can
linearly approximate your generative model for a specific point in a small neighborhood. 
"""

from .linear_methods import LatentScore, RadiiTrend, GaussianConvolutionAnalysis, IntrinsicDimensionScore

from .latent_statistics import ParallelogramLogCDF, EllipsoidCDFStatsCalculator, GaussianConvolutionStatsCalculator, GaussianConvolutionRateStatsCalculator

from .encoding_model import EncodingFlow, EncodingVAE

from .see_jacobian import SeeJacobian, IntrinsicDimensionFixing