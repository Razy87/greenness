"""
greenness
=========

Python reproduction of the functional-data + universal-kriging pipeline
described in:

    Hosseini, R. (2024). "Exploring Universal Kriging Modelling for
    Functional Greenness Data Captured by Digital Webcams."
    ICSTA 2024 Proceedings. https://www.avestia.com/ICSTA2024_Proceedings/files/paper/ICSTA_174.pdf

A parallel R implementation of the same pipeline lives in the `R/` folder.
This package covers the same steps with open-source Python equivalents:

    data       -> load & assemble the per-camera greenness curves
    smoothing  -> B-spline smoothing with GCV-selected penalty
    fpca       -> functional PCA + varimax rotation
    variogram  -> empirical variogram + model fitting (Matern / Exponential / Spherical / Gaussian)
    kriging    -> Universal Trace-Kriging (UTrK) and Universal (Co)Kriging of
                  rotated FPC scores (UCoK)
    evaluate   -> Monte-Carlo train/test RMSPE evaluation
"""

__version__ = "0.1.0"
