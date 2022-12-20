Dan Drennan final project for STAT647: Spatial Statistics at
Texas A&M University, College Station, Texas, USA. The project
reviews the work in Kidd and Katzfuss (2022, Bayesian Analysis)[^1].
The review presents their work as a climate emulation problem, reviews
past work in Gaussian processes, and presents their work as a covariance
selection problem. Additional topics covered in the review are neighborhood
selection using graphical modeling.

Code in this repository is nearly working but fails due to likelihoods always
evaluating to `nan` values. This prevents the model from being optimized. The
slides and paper folders can be compiled using LaTeX. Note there are figures
downloaded from the papers which will not render in the slides or paper without
being saved locally. Figure captions in both documents identify the original
figures referenced when reproducing captions.

This is a reupload of the original work to remove pycache files entirely from
the repository, so the version history is squashed entirely.

[^1] [https://doi.org/10.1214/21-BA1273](https://doi.org/10.1214/21-BA1273)
