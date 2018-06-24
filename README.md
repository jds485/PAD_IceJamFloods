# PAD_IceJamFloods

This repository provides supporting statistical analysis for the paper:
*Discussion of "Frequency of ice-jam flooding of Peace-Athabasca Delta" by S. Beltaos*
Code Repository Authors: Jared D. Smith (jds485@cornell.edu) and Jonathan R. Lamontagne (Jonathan.Lamontagne@tufts.edu)

If you have questions about the repository, please email Jared.

**Description of Repository**
This repository houses all of the necessary data and code to reproduce the statistical analyses presented in the above paper. All figures are provided, as well as intermediate figures and additional tests not presented in the paper. Note that some of these tests are not recommended by the authors.

The R script assumes that all files are located and written to one directory.
Flood data are loaded from the provided data file, and several cumulative flood count plots are made using that dataset.
Binomial exact tests and geometric distribution-based tests are used to compare to Beltaos' t-tests (but they are not appropriate tests to use because of nonstationarity in the probability of a flood over the period of record).
Beltaos' regression and Mann-Kendall test are reproduced (but these are inappropriate for reasons explained in the paper).
Pacific decadal osciallation indicators are plotted to see if climate may affect the flood frequency (not explored further here).
Autoregressive models are tested, but found to be nonstationary and are not recommended.
Block bootstrapping (preferred method) is used, and hypothesis tests are performed using the bootstrapped data. Additional tests not presented in the paper are provided for the maximum length of time before seeing a flood in the regulated era.
The Mann-Kendall test as used in Beltaos is applied to the bootstrapped data to illustrate the affect of stochastic variability on the results of the test.

**License**
See license file.

**Code Repository Citation**
Smith, J.D., and J.R. Lamontagne. (2018). PAD_IceJamFloods. Online GitHub Repository. https://github.com/jds485/PAD_IceJamFloods
