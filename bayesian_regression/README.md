# Readme for bayesian_regression folder

## Code
1. BayesianLogisticIJF.R - main R script that processes data and runs the Bayesian logistic regressions. This script runs in about 3.5 hours on 8 cores.
2. PlotProbsHistGCM.py - main python script that makes plots for historical and projected conditions. This script runs in a few minutes on 1 core.
3. utils.R - functions used in the main R script
4. mcmc_pairsEdits.R - edits the bayesplot::mcmc_pairs function to add x and y axis limits
5. WAIC_Edits.R - edits the WAIC function to have a second-order correction that is useful in small sample sizes 
6. CompareBetasLogisticBayes.py - script that compares the estimated regression model coefficients from the logistic regression bootstrap estimation from Lamontagne et al. (2021) in the logistic_regression folder and the Bayesian estimation of the same model in this folder.

## Data
1. cleaned_dataLMSAllYears.csv - 1915 to 2020 climatic variables and ice jam flood record for the Peace River. GP-BL Precip pct Avg: Grande Prarie/ Beaverlodge cumulative winter precipitation (snowpack); Fort Chip DDF: cumulative winter degree-days freezing at Fort Chipewyan, Fort Verm DDF: cumulative winter degree-days freezing at Fort Vermillion; Fort Smith: cumulative winter degree-days freezing at Fort Smith; MeltTest: melt test (days to reach a degree-days thaw threshold in spring); Flood: large = 1, not large = 0; AllLM: >= moderate = 1, else 0; AllLMS: >= small = 1, else 0; FloodMag: flood magnitude category.
2. GCM_Precip.csv, GCM_Temp_Chip.csv, GCM_Temp_Smith.csv, GCM_Temp_Verm.csv - downscaled GCM-RCP projections for Grande Prarie cumulative winter precipitation (snowpack), and cumulative winter degree-days freezing at Fort Chipewyan, Fort Smith, and Fort Vermillion.
