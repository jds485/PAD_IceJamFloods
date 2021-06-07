# PAD_IceJamFloods

This repository provides supporting statistical analysis for two papers:\
Timoney, K., J.D. Smith, J.R. Lamontagne, and M. Jasek. (2018). [*Discussion of "Frequency of ice-jam flooding of Peace-Athabasca Delta" by S. Beltaos*](https://doi.org/10.1139/cjce-2018-0409) Canadian Journal of Civil Engineering, 46(3).\
Lamontagne, J.R., M. Jasek, and J.D. Smith. (in revision for Cold Regions Science and Technology). *Coupling physical understanding and statistical modeling to estimate ice jam frequency under climate change*\

Code Repository Authors: Jared D. Smith (jared.d.smith485@gmail.com) and Jonathan R. Lamontagne (Jonathan.Lamontagne@tufts.edu)\
If you have questions about the repository, please email Jared.

# Description of Repository
This repository houses all of the necessary code to reproduce the statistical analyses presented in the above papers.\

## **block_bootstrap directory for Timoney et al. (2018)**
This directory corresponds to the Timoney et al. (2018) paper. All output data and figures are provided, as well as intermediate figures and additional statistical tests not presented in the paper. Note that some of these statistical tests are not recommended by the authors.\

R 3.5.0 was used for this analysis. The requirements.txt file provides the packages and versions used for this analysis. If needed, you can install these packages using `install.packages(c(<package1>, <package2>))`. If versions are different, you can use the devtools package function `install_version()`.\

The PAD_FloodAnalysis.R script assumes that all files are located and written to one directory.\
Flood data are loaded from the provided data file, and several cumulative flood count plots are made using that dataset.\
Binomial exact tests and geometric distribution-based tests are used to compare to Beltaos' t-tests (but they are not appropriate tests to use because of nonstationarity in the probability of a flood over the period of record).\

Beltaos' regression and Mann-Kendall test are reproduced (but these are inappropriate for reasons explained in the paper).\
Pacific decadal osciallation indicators are plotted to see if climate may affect the flood frequency (not explored further here).\
Autoregressive models are tested, but found to be nonstationary and are not recommended.\
Block bootstrapping (preferred method for this paper) is used, and hypothesis tests are performed using the bootstrapped data. Additional tests not presented in the paper are provided for the maximum length of time before seeing a flood in the regulated era.\
The Mann-Kendall test as used in Beltaos is applied to the bootstrapped data to illustrate the affect of stochastic variability on the results of the test.

## **logistic_regression directory for Lamontagne et al.**
This directory corresponds to the Lamontagne et al. paper. All paper figures are provided, as well as intermediate figures not presented in the paper.\
\
Python 3.7.4 was used for this analysis. The requirements.txt file provides the packages and versions used for this analysis. If needed, you can install these using `pip install -r requirements.txt`\

The utils.py and utils_figures.py scripts load user-defined functions that are used for this analysis. These functions depend on Jared's [firth_regression](https://gist.github.com/jds485/fd737a8314d45485f7e11f588baf88b9) function, which was forked and modified from John Lees' Gist. Click the provided link and place that function script into a directory of your choice. Place that directory name into the utils.py script where os.chdir is located.\

The example.py script assumes that all files are written to the .\PAD_IceJamFloods\logistic_regression directory. This script takes up to 5 hours to run. The resulting data file is about 39 GB. We estimate that a machine with 16 GB RAM or more is required to run this script. This script will generate many figures to the plotting window, some of which were not used in the paper. We saved the final paper figures in the PaperFigures directory.

# License
See license file.

# Code Repository Citation
Smith, J.D., and J.R. Lamontagne. (2018). PAD_IceJamFloods. Online GitHub Repository. https://github.com/jds485/PAD_IceJamFloods
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4474826.svg)](https://doi.org/10.5281/zenodo.4474826)
