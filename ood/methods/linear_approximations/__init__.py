"""
This package contains all the OOD methods that rely on the fact that you can
linearly approximate your generative model for a specific point in a small neighborhood. 
"""

from .methods import CDFScore, CDFTrend