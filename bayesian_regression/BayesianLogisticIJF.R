#Bayesian Logistic Regression Model of Peace Athabasca Delta Large Ice Jam Floods
# The working directory should be the location of this R script:
# "LogisticPADIJF\\PAD_IceJamFloods\\bayesian_regression"

#Load Libraries----
library(logistf)
library(BayesianTools)
library(parallel)
library(foreach)
library(iterators)
library(doParallel)
library(bayesplot)
library(rlang)
library(coda)
library(ggplot2)
library(loo)

#User-defined functions----
source("utils.R")

#Load edited functions from code repos----
source("mcmc_pairsEdits.R")
source("WAIC_Edits.R")

#Load and process data----
Data = read.csv(file = 'cleaned_dataLMSAllYears.csv', stringsAsFactors = FALSE)
#Remove dam filling years
Data = Data[-which(Data$Year %in% 1968:1971),]
#Set X (predictors) and Y (flood indicators), and make a separate year vector for plotting
#Large floods
Y = Data$Flood
#Large, Medium, Small, and Unknown floods, for plotting
Y_All = Data$AllLMS
X = Data[,-c(1,7,8,9,10)]
years = Data$Year
FloodMag = Data$FloodMag

# Select years based on variables that have data----
#Large floods, 1962-2020 (used in Lamontagne et al. logistic regression paper)
years_hold = years[years > 1961]
Y_hold = Y[years > 1961]
X_hold = X[years > 1961,]
#Large floods, 1962-2020, only X variables with data in that range + Fort Smith
tmp_rm = apply(X = is.na(as.matrix(X_hold[,c(1,2,3,4)])), MARGIN = 1, FUN = any)
years_holdsm = years_hold[!tmp_rm]
Y_holdsm = Y_hold[!tmp_rm]
X_holdsm = X_hold[!tmp_rm,]
#Large floods, 1915-2020, only X variables with data in that range
tmp_rm = apply(X = is.na(as.matrix(X[,c(1,2,4)])), MARGIN = 1, FUN = any)
years_L15 = years[!tmp_rm]
Y_L15 = Y[!tmp_rm]
X_L15 = X[!tmp_rm,]
FloodMag_L15 = FloodMag[!tmp_rm]
#Large floods, 1915-2020, only X variables with data in that range + Fort Smith
tmp_rm = apply(X = is.na(as.matrix(X[,c(1,2,3,4)])), MARGIN = 1, FUN = any)
years_L15sm = years[!tmp_rm]
Y_L15sm = Y[!tmp_rm]
X_L15sm = X[!tmp_rm,]
FloodMag_L15sm = FloodMag[!tmp_rm]
#All floods, 1915-2020, only X variables with data in that range
tmp_rm = apply(X = is.na(as.matrix(X[,c(1,2,4)])), MARGIN = 1, FUN = any)
years_All15 = years[!tmp_rm]
Y_All15 = Y_All[!tmp_rm]
X_All15 = X[!tmp_rm,]
FloodMag_All15 = FloodMag[!tmp_rm]
#All floods, 1915-2020, only X variables with data in that range + Fort Smith
tmp_rm = apply(X = is.na(as.matrix(X[,c(1,2,3,4)])), MARGIN = 1, FUN = any)
years_All15sm = years[!tmp_rm]
Y_All15sm = Y_All[!tmp_rm]
X_All15sm = X[!tmp_rm,]
FloodMag_All15sm = FloodMag[!tmp_rm]
rm(tmp_rm)

# Normalize X and add a constant----
X_hold = normalize(X_hold)
X_All15 = normalize(X_All15)
X_All15sm = normalize(X_All15sm)
X_L15 = normalize(X_L15)
X_L15sm = normalize(X_L15sm)
#Want to use the same normalization as the 1915-2020 data to compare resulting marginal distrbutions
#for the principal components.
X_holdsm = X_L15sm[years_L15sm >= 1962,]

#Exploratory Data Analysis----
dir.create(path = 'EDA', showWarnings = FALSE)
# Ft Chip----
plot_EDA_AllFloods(fname = 'EDA/EDA_GPBLPrecip+FtChipDDF_AllFloods.png', X_All15, 
                   X_colname = 'Fort.Chip.DDF', X_label = 'Fort Chip. DDF', 
                   Y = Y_All15, FloodMag = FloodMag_All15)

plot_EDA_MSFloodsPre1962('EDA/EDA_GPBLPrecip+FtChipDDF_MSFloodsPre1962.png', X_All15, 
                         X_colname = 'Fort.Chip.DDF', X_label = 'Fort Chip. DDF', 
                         Y = Y_All15, FloodMag = FloodMag_All15, years = years_All15)

# Ft Verm----
plot_EDA_AllFloods(fname = 'EDA/EDA_GPBLPrecip+FtVermDDF_AllFloods.png', X_All15, 
                   X_colname = 'Fort.Verm.DDF', X_label = 'Fort Verm. DDF', 
                   Y = Y_All15, FloodMag = FloodMag_All15)

plot_EDA_MSFloodsPre1962('EDA/EDA_GPBLPrecip+FtVermDDF_MSFloodsPre1962.png', X_All15, 
                         X_colname = 'Fort.Verm.DDF', X_label = 'Fort Verm. DDF', 
                         Y = Y_All15, FloodMag = FloodMag_All15, years = years_All15)


# Ft Smith----
plot_EDA_AllFloods(fname = 'EDA/EDA_GPBLPrecip+FtSmithDDF_AllFloods.png', X_All15sm, 
                   X_colname = 'Fort.Smith', X_label = 'Fort Smith DDF', 
                   Y = Y_All15sm, FloodMag = FloodMag_All15sm)

plot_EDA_MSFloodsPre1962('EDA/EDA_GPBLPrecip+FtSmithDDF_MSFloodsPre1962.png', X_All15sm, 
                         X_colname = 'Fort.Smith', X_label = 'Fort Smith DDF', 
                         Y = Y_All15sm, FloodMag = FloodMag_All15sm, years = years_All15sm)


# PCA of data Chip, Verm, Smith, Precip----
PCAcvs_All15 = prcomp(x = X_All15sm[,c(2,3,4,5)])
PCAcvs_All15$varRatio = PCAcvs_All15$sdev^2 / sum(PCAcvs_All15$sdev^2)
plot_EDA_PCA_MSFloodsPre1962_color(fname = 'EDA/PCAcvsp_MSFloodsPre1962_Color.png', X = PCAcvs_All15,
                                   X_label = 'PC1 - Mostly Winter Degree-Days Freezing (81% variance)',
                                   Y_label = 'PC2 - Mostly Winter Snowpack (15% variance)', 
                                   Y = Y_All15sm, FloodMag = FloodMag_All15sm, years = years_All15sm)

plot_EDA_PCA_MSFloodsPre1962_fill(fname = 'EDA/PCAcvsp_MSFloodsPre1962_Fill.png', X = PCAcvs_All15,
                                   X_label = 'PC1 - Mostly Winter Degree-Days Freezing (81% variance)',
                                   Y_label = 'PC2 - Mostly Winter Snowpack (15% variance)', 
                                   Y = Y_All15sm, FloodMag = FloodMag_All15sm, years = years_All15sm)

#  Figure 2----
plot_EDA_PCA_MSFloodsPre1962_fill(fname = 'EDA/Fig2-PCA_fill.pdf', X = PCAcvs_All15,
                                  X_label = 'PC1 - Mostly Winter Degree-Days Freezing (81% variance)',
                                  Y_label = 'PC2 - Mostly Winter Snowpack (15% variance)', 
                                  Y = Y_All15sm, FloodMag = FloodMag_All15sm, years = years_All15sm, pdf = TRUE)

#  Labeled uncertain years----
plot_EDA_PCA_MSFloodsPre1962_fill(fname = 'EDA/PCA_MSPre1962_LargeLabeled.png', X = PCAcvs_All15,
                                  X_label = 'PC1 - Mostly Winter Degree-Days Freezing (81% variance)',
                                  Y_label = 'PC2 - Mostly Winter Snowpack (15% variance)', 
                                  Y = Y_All15sm, FloodMag = FloodMag_All15sm, years = years_All15sm, label_mag = 'L')

plot_EDA_PCA_MSFloodsPre1962_fill(fname = 'EDA/PCA_MSPre1962_MedLabeled.png', X = PCAcvs_All15,
                                  X_label = 'PC1 - Mostly Winter Degree-Days Freezing (81% variance)',
                                  Y_label = 'PC2 - Mostly Winter Snowpack (15% variance)', 
                                  Y = Y_All15sm, FloodMag = FloodMag_All15sm, years = years_All15sm, label_mag = 'M')

plot_EDA_PCA_MSFloodsPre1962_fill(fname = 'EDA/PCA_MSPre1962_SmallLabeled.png', X = PCAcvs_All15,
                                  X_label = 'PC1 - Mostly Winter Degree-Days Freezing (81% variance)',
                                  Y_label = 'PC2 - Mostly Winter Snowpack (15% variance)', 
                                  Y = Y_All15sm, FloodMag = FloodMag_All15sm, years = years_All15sm, label_mag = 'S')

plot_EDA_PCA_MSFloodsPre1962_fill(fname = 'EDA/PCA_MSPre1962_UnknownLabeled.png', X = PCAcvs_All15,
                                  X_label = 'PC1 - Mostly Winter Degree-Days Freezing (81% variance)',
                                  Y_label = 'PC2 - Mostly Winter Snowpack (15% variance)', 
                                  Y = Y_All15sm, FloodMag = FloodMag_All15sm, years = years_All15sm, label_mag = 'U')

plot_EDA_PCA_MSFloodsPre1962_fill(fname = 'EDA/PCA_MSPre1962_NoFloodLabeled.png', X = PCAcvs_All15,
                                  X_label = 'PC1 - Mostly Winter Degree-Days Freezing (81% variance)',
                                  Y_label = 'PC2 - Mostly Winter Snowpack (15% variance)', 
                                  Y = Y_All15sm, FloodMag = FloodMag_All15sm, years = years_All15sm, label_mag = 'N')

#Firth Logistic Regression Model----
# Best model from Lamontagne et al. 1962-2020 ----
res_Firth = logistf(formula = Y_hold ~ X_hold$Fort.Verm.DDF + X_hold$GP.BL.Precip.pct.Avg, 
                    pl = TRUE, alpha = 0.05, firth = TRUE)
res_Firth$aic = aic(res_Firth)
res_Firth$aicc = aicc(res_Firth)
#  Interaction model----
res_Firth_cross = logistf(formula = Y_hold ~ X_hold$Fort.Verm.DDF * X_hold$GP.BL.Precip.pct.Avg, 
                          pl = TRUE, alpha = 0.05, firth = TRUE)
res_Firth_cross$aic = aic(res_Firth_cross)
res_Firth_cross$aicc = aicc(res_Firth_cross)
#  Interaction model with Melt Test----
res_Firth_Meltcross = logistf(formula = Y_hold ~ I(X_hold$MeltTest * X_hold$Fort.Verm.DDF) + 
                                X_hold$GP.BL.Precip.pct.Avg, 
                              pl = TRUE, alpha = 0.05, firth = TRUE)
res_Firth_Meltcross$aic = aic(res_Firth_Meltcross)
res_Firth_Meltcross$aicc = aicc(res_Firth_Meltcross)

#  PCA - Chip, Verm, Smith, Precip----
#using PCA with all available data
predPCAcvs_All15 = predict(PCAcvs_All15)
res_Firth_sm = logistf(formula = Y_holdsm ~ predict(PCAcvs_All15, X_holdsm)[,1] + predict(PCAcvs_All15, X_holdsm)[,2], 
                       pl = TRUE, alpha = 0.05, firth = TRUE)
res_Firth_sm$aic = aic(res_Firth_sm)
res_Firth_sm$aicc = aicc(res_Firth_sm)

# Large floods 1915-2020 (treated as 1, unknown mag = 0, no uncertainty in either)----
#  Testing Precip and Additive Precip + Temp----
res_Firth_L15_Prcp = logistf(formula = Y_L15 ~ X_L15$GP.BL.Precip.pct.Avg, pl = TRUE, alpha = 0.05, firth = TRUE)
res_Firth_L15_Prcp$aic = aic(res_Firth_L15_Prcp)
res_Firth_L15_Prcp$aicc = aicc(res_Firth_L15_Prcp)
res_Firth_L15 = logistf(formula = Y_L15 ~ X_L15$Fort.Verm.DDF + X_L15$GP.BL.Precip.pct.Avg, pl = TRUE, alpha = 0.05, firth = TRUE)
res_Firth_L15$aic = aic(res_Firth_L15)
res_Firth_L15$aicc = aicc(res_Firth_L15)
res_Firth_L15Cp = logistf(formula = Y_L15 ~ X_L15$Fort.Chip.DDF + X_L15$GP.BL.Precip.pct.Avg, pl = TRUE, alpha = 0.05, firth = TRUE)
res_Firth_L15Cp$aic = aic(res_Firth_L15Cp)
res_Firth_L15Cp$aicc = aicc(res_Firth_L15Cp)
#  PCA - Chip, Verm, Smith, Precip----
PCAcvs = prcomp(x = X_L15sm[,c(2,3,4,5)])
PCAcvs$varRatio = PCAcvs$sdev^2 / sum(PCAcvs$sdev^2)
#Same as predPCAcvs_All15
predPCAcvs = predict(PCAcvs)
res_Firth_L15_PC3 = logistf(formula = Y_L15sm ~ predict(PCAcvs)[,1] + predict(PCAcvs)[,2], 
                            pl = TRUE, alpha = 0.05, firth = TRUE)
res_Firth_L15_PC3$aic = aic(res_Firth_L15_PC3)
res_Firth_L15_PC3$aicc = aicc(res_Firth_L15_PC3)
#  PCA - Chip, Verm, Precip - same time period as Ft. Smith----
PCAcvst = prcomp(x = X_L15sm[,c(2,3,5)])
PCAcvst$varRatio = PCAcvst$sdev^2 / sum(PCAcvst$sdev^2)
res_Firth_L15_PC3t = logistf(formula = Y_L15sm ~ predict(PCAcvst)[,1] + predict(PCAcvst)[,2], 
                             pl = TRUE, alpha = 0.05, firth = TRUE)
res_Firth_L15_PC3t$aic = aic(res_Firth_L15_PC3t)
res_Firth_L15_PC3t$aicc = aicc(res_Firth_L15_PC3t)
#  PCA - Chip, Verm, Precip----
PCA = prcomp(x = X_L15[,c(2,3,5)])
PCA$varRatio = PCA$sdev^2 / sum(PCA$sdev^2)
res_Firth_L15_PC2 = logistf(formula = Y_L15 ~ predict(PCA)[,1] + predict(PCA)[,2], pl = TRUE, alpha = 0.05, firth = TRUE)
res_Firth_L15_PC2$aic = aic(res_Firth_L15_PC2)
res_Firth_L15_PC2$aicc = aicc(res_Firth_L15_PC2)
#  PCA- FtVerm----
PCAv = prcomp(x = X_L15[,c(3,5)])
PCAv$varRatio = PCAv$sdev^2 / sum(PCAv$sdev^2)
res_Firth_L15_PC1 = logistf(formula = Y_L15 ~ predict(PCAv)[,1] + predict(PCAv)[,2], pl = TRUE, alpha = 0.05, firth = TRUE)
res_Firth_L15_PC1$aic = aic(res_Firth_L15_PC1)
res_Firth_L15_PC1$aicc = aicc(res_Firth_L15_PC1)
#  PCA - Chip, Verm, Smith, Melt, Precip----
PCAcvsm_L15 = prcomp(x = X_L15sm[,c(2,3,4,5,7)])
PCAcvsm_L15$varRatio = PCAcvsm_L15$sdev^2 / sum(PCAcvsm_L15$sdev^2)
res_Firth_L15_PC4 = logistf(formula = Y_L15sm ~ predict(PCAcvsm_L15)[,1] + predict(PCAcvsm_L15)[,2] + predict(PCAcvsm_L15)[,3], pl = TRUE, alpha = 0.05, firth = TRUE)
res_Firth_L15_PC4$aic = aic(res_Firth_L15_PC4)
res_Firth_L15_PC4$aicc = aicc(res_Firth_L15_PC4)

#Best model of large floods only is PCA with Ft verm, chip, smith DDF + GP/BL precip.

#MCMC Parameter Estimation Method----
#Large Floods, 1962 - 2020, Same Parameters as Lamontagne et al.----
# Prior Definition----
#MVN Centered on Firth MAP for non-PCA model
prior = createTruncatedNormalPrior(mean = as.numeric(res_Firth$coefficients), sd = c(10,10,10), 
                                   lower = -Inf, upper = Inf)
#MVN Centered on 0
prior2 = createTruncatedNormalPrior(mean = c(0,0,0), sd = c(10,10,10), lower = -Inf, upper = Inf)
#Uniform
prior3 = createUniformPrior(lower = c(-30,-30,-30), upper = c(30,30,30))

# Likelihood Definition----
likelihood <- function(param, sum = TRUE){
  #Firth log-likelihood - equals the same as summed pointwise ll below
  #ll = tryCatch(logistf.fit(x = as.matrix(X_hold[,c(1,3,5)]), y = Y_hold, weight = NULL, 
  #                          offset = NULL, firth = TRUE, col.fit = 0, init = param, 
  #                          control = logistf.control())$loglik,
  #              error = function(a){return(-Inf)}, finally = '')
  
  #point likelihoods
  Xi = as.matrix(X_hold[,c(1,3,5)])
  #numerator
  num = exp(param[1] + param[2]*Xi[,2] + param[3]*Xi[,3])
  #y-hat
  pred = num/(1+num)
  #weights
  w = diag(pred * (1-pred))
  #Information matrix
  I = t(Xi) %*% w %*% Xi
  pll = vector('numeric', length=length(Y_hold))
  for (j in 1:length(Y_hold)){
    pll[j] = Y_hold[j]*log(pred[j]) + (1 - Y_hold[j])*log(1 - pred[j])
  }
  #sum with Firth correction
  ll = sum(pll) + 0.5*determinant(I, logarithm = TRUE)$modulus[1]
  if (is.nan(ll)){
    ll = -Inf
  }
  
  #Return sum of the log-likelihoods
  return(if (sum == TRUE) ll else (pll + 0.5*determinant(I, logarithm = TRUE)$modulus[1]/length(Y_hold))) 
}

#Fort smith data
likelihood_sm <- function(param, sum = TRUE){
  #point likelihoods
  Xi = as.matrix(cbind(X_holdsm[,1], predict(PCAcvs, X_holdsm)[,c(1,2)]))
  #numerator
  num = exp(param[1] + param[2]*Xi[,2] + param[3]*Xi[,3])
  #y-hat
  pred = num/(1+num)
  #weights
  w = diag(pred * (1-pred))
  #Information matrix
  I = t(Xi) %*% w %*% Xi
  pll = vector('numeric', length=length(Y_holdsm))
  for (j in 1:length(Y_holdsm)){
    pll[j] = Y_holdsm[j]*log(pred[j]) + (1 - Y_holdsm[j])*log(1 - pred[j])
  }
  #sum with Firth correction
  ll = sum(pll) + 0.5*determinant(I, logarithm = TRUE)$modulus[1]
  if (is.nan(ll)){
    ll = -Inf
  }
  
  #Return sum of the log-likelihoods
  return(if (sum == TRUE) ll else (pll + 0.5*determinant(I, logarithm = TRUE)$modulus[1]/length(Y_holdsm))) 
}

#Fort smith data, 1915-2020
likelihood_sm_1915 <- function(param, sum = TRUE){
  #point likelihoods
  Xi = as.matrix(cbind(X_L15sm[,1], predict(PCAcvs, X_L15sm)[,c(1,2)]))
  #numerator
  num = exp(param[1] + param[2]*Xi[,2] + param[3]*Xi[,3])
  #y-hat
  pred = num/(1+num)
  #weights
  w = diag(pred * (1-pred))
  #Information matrix
  I = t(Xi) %*% w %*% Xi
  pll = vector('numeric', length=length(Y_L15sm))
  for (j in 1:length(Y_L15sm)){
    pll[j] = Y_L15sm[j]*log(pred[j]) + (1 - Y_L15sm[j])*log(1 - pred[j])
  }
  #sum with Firth correction
  ll = sum(pll) + 0.5*determinant(I, logarithm = TRUE)$modulus[1]
  if (is.nan(ll)){
    ll = -Inf
  }
  
  #Return sum of the log-likelihoods
  return(if (sum == TRUE) ll else (pll + 0.5*determinant(I, logarithm = TRUE)$modulus[1]/length(Y_L15sm))) 
}

# MCMC Setup----
settingsDREAMzs_1 = list(iterations = 330000, gamma= NULL, eps = 0, e = 0.05, parallel = NULL, Z = NULL, 
                         ZupdateFrequency = 30, pSnooker = 0.1, DEpairs = 2,
                         nCR = 3, pCRupdate = TRUE, updateInterval = 10,
                         #burnin must be >= adaptation.
                         burnin = 30000, adaptation = 30000, thin = 30, message = FALSE, startValue = 7)

# Posterior Solvers----
#  DREAMzs: prior #1 - normal, Firth mean----
dir.create(path = 'DREAMzs1', showWarnings = FALSE)
setUpDREAMzs_1 <- createBayesianSetup(likelihood, prior = prior, parallel = 7, 
                                      parallelOptions = list(packages=list('BayesianTools'), 
                                                             variables=list('X_hold','Y_hold'), dlls=NULL), 
                                      names = c('Int', 'Ft.Verm DDF', 'GP/BL Precip.'), 
                                      plotLower = c(-15, -6, 0), plotUpper = c(0, 2, 6), 
                                      plotBest = as.numeric(res_Firth$coefficients))
set.seed(16421)
outDREAMzs_1 <- runMCMC(bayesianSetup = setUpDREAMzs_1, sampler = "DREAMzs", settings = settingsDREAMzs_1)
#Remove all -Inf in chain. These result from poor initial conditions (prior samples)
outDREAMzs_1 = removeInf(outDREAMzs_1)
#Check for remaining infinities
if(any(outDREAMzs_1$InfCk > 0)){
  print('infinities remain within MCMC chain posteriors')
}
summary(outDREAMzs_1)

#Compute WAIC and DIC
outDREAMzs_1$WAICBT = WAIC(outDREAMzs_1, numSamples = 100000)
outDREAMzs_1$WAIC2BT = WAIC2(outDREAMzs_1, numSamples = 100000)
outDREAMzs_1$DIC = DIC(outDREAMzs_1)
#Pointwise log-likelihoods
outDREAMzs_1$pll = outDREAMzs_1$setup$likelihood$density(
  getSample(outDREAMzs_1, parametersOnly = F)[,1:outDREAMzs_1$setup$numPars],
  sum = FALSE)
#LOO
outDREAMzs_1$LOO = loo(outDREAMzs_1$pll, cores = 7, save_psis = TRUE, 
                       r_eff = relative_eff(x = exp(outDREAMzs_1$pll), 
                                            chain_id = c(rep(1,nrow(outDREAMzs_1$chain[[1]])), 
                                                         rep(2,nrow(outDREAMzs_1$chain[[1]])), 
                                                         rep(3,nrow(outDREAMzs_1$chain[[1]])), 
                                                         rep(4,nrow(outDREAMzs_1$chain[[1]])), 
                                                         rep(5,nrow(outDREAMzs_1$chain[[1]])), 
                                                         rep(6,nrow(outDREAMzs_1$chain[[1]])), 
                                                         rep(7,nrow(outDREAMzs_1$chain[[1]])))))
outDREAMzs_1$WAIC = waic(x = outDREAMzs_1$pll)
outDREAMzs_1$WAIC2 = waic2(x = outDREAMzs_1$pll)
#Rejection rate
outDREAMzs_1$rejectRate = rejectionRate(outDREAMzs_1$chain)
#Highest Posterior Density Intervals
outDREAMzs_1$HPD = HPDinterval(outDREAMzs_1$chain)
#Credible Intervals
outDREAMzs_1$CI = getCredibleIntervals(outDREAMzs_1$chain[[1]])
#Effective sample size for each parameter should be similar
outDREAMzs_1$Neff = effectiveSize(outDREAMzs_1$chain[,1:3])

#Sample only the last 1000 from each chain
sample_chain_inds = (nrow(outDREAMzs_1$chain[[1]])-1000):nrow(outDREAMzs_1$chain[[1]])

png('DREAMzs1/traceplot2.png', res = 300, units = 'in', width = 7, height = 7)
mcmc_trace(x = outDREAMzs_1$chain[,1:3], size = 0.05)
dev.off()

png('DREAMzs1/correlplot.png', res = 300, units = 'in', width = 7, height = 7)
correlationPlot(outDREAMzs_1$chain[sample_chain_inds,1:3], method = 'spearman')
dev.off()

png('DREAMzs1/gelmanplot.png', res = 300, units = 'in', width = 7, height = 7)
gelmanDiagnostics(outDREAMzs_1, plot = TRUE, thin = 0)
dev.off()

png('DREAMzs1/marginalplot.png', res = 300, units = 'in', width = 7, height = 7)
marginalPlot(x = outDREAMzs_1$chain[sample_chain_inds, 1:3], prior = prior$sampler(n = 50000), singlePanel = FALSE, 
             xrange = t(rbind(c(-20,5),c(-10,5),c(-5,15))))
dev.off()

png('DREAMzs1/PairPlot.png', res = 300, units = 'in', width = 7, height = 7)
mcmc_pairs(outDREAMzs_1$chain[sample_chain_inds, 1:3], diag_fun = 'dens', off_diag_fun = 'scatter', 
           off_diag_args = list(size=0.5,alpha=0.5), 
           xlim = rbind(c(-15,0),c(-6,1),c(0,6),c(-15,0),c(-6,1),c(0,6),c(-15,0),c(-6,1),c(0,6)),
           ylim = rbind(c(0,NA),c(-15,0),c(-15,0),c(-6,0),c(0,NA),c(-6,0),c(0,6),c(0,6),c(0,NA)))
dev.off()

png('DREAMzs1/autocorrplot.png', res = 300, units = 'in', width = 7, height = 7)
autocorr.plot(x = outDREAMzs_1$chain[[1]][-1,1:3], lag.max = 100)
dev.off()

png('DREAMzs1/autocorrplot2.png', res = 300, units = 'in', width = 7, height = 7)
mcmc_acf(x = outDREAMzs_1$chain, pars = outDREAMzs_1$setup$names, lags = 20)
dev.off()

png('DREAMzs1/densOverlay.png', res = 300, units = 'in', width = 7, height = 7)
mcmc_dens_overlay(outDREAMzs_1$chain[sample_chain_inds,])
dev.off()

png('DREAMzs1/HPD.png', res = 300, units = 'in', width = 7, height = 7)
mcmc_areas(x = outDREAMzs_1$chain[sample_chain_inds,], pars = outDREAMzs_1$setup$names, 
           point_est = 'mean', prob = 0.95)
dev.off()

png('DREAMzs1/HPD_boxplot.png', res = 300, units = 'in', width = 7, height = 7)
mcmc_intervals(x = outDREAMzs_1$chain[sample_chain_inds,], pars = outDREAMzs_1$setup$names,
               prob_outer = 0.95)
dev.off()

geweke.plot(x = outDREAMzs_1$chain[,1:3],ask = FALSE, nbins = 8)

stopParallel(setUpDREAMzs_1)

#  DREAMzs: prior #2 - normal, 0 mean----
dir.create(path = 'DREAMzs_L15_VermPrecip_1962-2020', showWarnings = FALSE)
setUpDREAMzs_1p <- createBayesianSetup(likelihood, prior = prior2, parallel = 7, 
                                      parallelOptions = list(packages=list('BayesianTools'), 
                                                             variables=list('X_hold','Y_hold'), dlls=NULL), 
                                      names = c('Int', 'Ft.Verm DDF', 'GP/BL Precip.'), 
                                      plotLower = c(-15, -6, 0), plotUpper = c(0, 2, 6), 
                                      plotBest = as.numeric(res_Firth$coefficients))
set.seed(16421)
outDREAMzs_1p <- runMCMC(bayesianSetup = setUpDREAMzs_1p, sampler = "DREAMzs", settings = settingsDREAMzs_1)
#Remove all -Inf in chain. These result from poor initial conditions (prior samples)
outDREAMzs_1p = removeInf(outDREAMzs_1p)
#Check for remaining infinities
if(any(outDREAMzs_1p$InfCk > 0)){
  print('infinities remain within MCMC chain posteriors')
}
summary(outDREAMzs_1p)

#Compute WAIC and DIC
outDREAMzs_1p$WAICBT = WAIC(outDREAMzs_1p, numSamples = 100000)
outDREAMzs_1p$WAIC2BT = WAIC2(outDREAMzs_1p, numSamples = 100000)
outDREAMzs_1p$DIC = DIC(outDREAMzs_1p)
#Pointwise log-likelihoods
outDREAMzs_1p$pll = outDREAMzs_1p$setup$likelihood$density(getSample(outDREAMzs_1p, parametersOnly = F)[,1:outDREAMzs_1p$setup$numPars], sum = FALSE)
#LOO
outDREAMzs_1p$LOO = loo(outDREAMzs_1p$pll, cores = 7, save_psis = TRUE, 
                        r_eff = relative_eff(x = exp(outDREAMzs_1p$pll), 
                                             chain_id = c(rep(1,nrow(outDREAMzs_1p$chain[[1]])), 
                                                          rep(2,nrow(outDREAMzs_1p$chain[[1]])), 
                                                          rep(3,nrow(outDREAMzs_1p$chain[[1]])), 
                                                          rep(4,nrow(outDREAMzs_1p$chain[[1]])), 
                                                          rep(5,nrow(outDREAMzs_1p$chain[[1]])), 
                                                          rep(6,nrow(outDREAMzs_1p$chain[[1]])), 
                                                          rep(7,nrow(outDREAMzs_1p$chain[[1]])))))
outDREAMzs_1p$WAIC = waic(x = outDREAMzs_1p$pll)
outDREAMzs_1p$WAIC2 = waic2(x = outDREAMzs_1p$pll)
#Rejection rate
outDREAMzs_1p$rejectRate = rejectionRate(outDREAMzs_1p$chain)
#Highest Posterior Density Intervals
outDREAMzs_1p$HPD = HPDinterval(outDREAMzs_1p$chain)
#Credible Intervals
outDREAMzs_1p$CI = getCredibleIntervals(outDREAMzs_1p$chain[[1]])
#Effective sample size for each parameter should be similar
outDREAMzs_1p$Neff = effectiveSize(outDREAMzs_1p$chain[,1:3])

#Sample only the last 1000 from each chain
sample_chain_inds = (nrow(outDREAMzs_1p$chain[[1]])-1000):nrow(outDREAMzs_1p$chain[[1]])

dir.create('DREAMzs_L15_VermPrecip_1962-2020/MCMC', showWarnings = FALSE)
png('DREAMzs_L15_VermPrecip_1962-2020/MCMC/traceplot2.png', res = 300, units = 'in', width = 7, height = 7)
mcmc_trace(x = outDREAMzs_1p$chain[,1:3], size = 0.05)
dev.off()

png('DREAMzs_L15_VermPrecip_1962-2020/MCMC/correlplot.png', res = 300, units = 'in', width = 7, height = 7)
correlationPlot(outDREAMzs_1p$chain[sample_chain_inds,1:3], method = 'spearman')
dev.off()

png('DREAMzs_L15_VermPrecip_1962-2020/MCMC/gelmanplot.png', res = 300, units = 'in', width = 7, height = 7)
gelmanDiagnostics(outDREAMzs_1p, plot = TRUE, thin = 0)
dev.off()

png('DREAMzs_L15_VermPrecip_1962-2020/MCMC/marginalplot.png', res = 300, units = 'in', width = 7, height = 7)
marginalPlot(x = outDREAMzs_1p$chain[sample_chain_inds,1:3], prior = prior2$sampler(n = 50000), singlePanel = FALSE, 
             xrange = t(rbind(c(-20,5),c(-10,5),c(-5,15))))
dev.off()

png('DREAMzs_L15_VermPrecip_1962-2020/MCMC/PairPlot.png', res = 300, units = 'in', width = 7, height = 7)
mcmc_pairs(outDREAMzs_1p$chain[sample_chain_inds,1:3], diag_fun = 'dens', off_diag_fun = 'scatter', 
           off_diag_args = list(size=0.5,alpha=0.5), 
           xlim = rbind(c(-15,0),c(-6,1),c(0,6),c(-15,0),c(-6,1),c(0,6),c(-15,0),c(-6,1),c(0,6)),
           ylim = rbind(c(0,NA),c(-15,0),c(-15,0),c(-6,0),c(0,NA),c(-6,0),c(0,6),c(0,6),c(0,NA)))
dev.off()

png('DREAMzs_L15_VermPrecip_1962-2020/MCMC/autocorrplot.png', res = 300, units = 'in', width = 7, height = 7)
autocorr.plot(x = outDREAMzs_1p$chain[[1]][-1,1:3], lag.max = 100)
dev.off()

png('DREAMzs_L15_VermPrecip_1962-2020/MCMC/autocorrplot2.png', res = 300, units = 'in', width = 7, height = 7)
mcmc_acf(x = outDREAMzs_1p$chain, pars = outDREAMzs_1p$setup$names, lags = 20)
dev.off()

png('DREAMzs_L15_VermPrecip_1962-2020/MCMC/densOverlay.png', res = 300, units = 'in', width = 7, height = 7)
mcmc_dens_overlay(outDREAMzs_1p$chain[sample_chain_inds,])
dev.off()

png('DREAMzs_L15_VermPrecip_1962-2020/MCMC/HPD.png', res = 300, units = 'in', width = 7, height = 7)
mcmc_areas(x = outDREAMzs_1p$chain[sample_chain_inds,], pars = outDREAMzs_1p$setup$names, 
           point_est = 'mean', prob = 0.95)
dev.off()

png('DREAMzs_L15_VermPrecip_1962-2020/MCMC/HPD_boxplot.png', res = 300, units = 'in', width = 7, height = 7)
mcmc_intervals(x = outDREAMzs_1p$chain[sample_chain_inds,], pars = outDREAMzs_1p$setup$names,
               prob_outer = 0.95)
dev.off()

geweke.plot(x = outDREAMzs_1p$chain[,1:3], ask = FALSE, nbins = 8)

stopParallel(setUpDREAMzs_1p)

#  DREAMzs: prior #3 - uniform----
dir.create(path = 'DREAMzs3p', showWarnings = FALSE)
setUpDREAMzs_3p <- createBayesianSetup(likelihood, prior = prior3, parallel = 7, 
                                       parallelOptions = list(packages=list('BayesianTools'), 
                                                              variables=list('X_hold','Y_hold'), dlls=NULL), 
                                       names = c('Int', 'Ft.Verm DDF', 'GP/BL Precip.'), 
                                       plotLower = c(-15, -6, 0), plotUpper = c(0, 2, 6), 
                                       plotBest = as.numeric(res_Firth$coefficients))
set.seed(16421)
outDREAMzs_3p <- runMCMC(bayesianSetup = setUpDREAMzs_3p, sampler = "DREAMzs", settings = settingsDREAMzs_1)
#Remove all -Inf in chain. These result from poor initial conditions (prior samples)
outDREAMzs_3p = removeInf(outDREAMzs_3p)
#Check for remaining infinities
if(any(outDREAMzs_3p$InfCk > 0)){
  print('infinities remain within MCMC chain posteriors')
}
summary(outDREAMzs_3p)

#Compute WAIC and DIC
outDREAMzs_3p$WAICBT = WAIC(outDREAMzs_3p, numSamples = 100000)
outDREAMzs_3p$WAIC2BT = WAIC2(outDREAMzs_3p, numSamples = 100000)
outDREAMzs_3p$DIC = DIC(outDREAMzs_3p)
#Pointwise log-likelihoods
outDREAMzs_3p$pll = outDREAMzs_3p$setup$likelihood$density(getSample(outDREAMzs_3p, parametersOnly = F)[,1:outDREAMzs_3p$setup$numPars], sum = FALSE)
#LOO
outDREAMzs_3p$LOO = loo(outDREAMzs_3p$pll, cores = 7, save_psis = TRUE, 
                        r_eff = relative_eff(x = exp(outDREAMzs_3p$pll), 
                                             chain_id = c(rep(1,nrow(outDREAMzs_3p$chain[[1]])), 
                                                          rep(2,nrow(outDREAMzs_3p$chain[[1]])), 
                                                          rep(3,nrow(outDREAMzs_3p$chain[[1]])),
                                                          rep(4,nrow(outDREAMzs_3p$chain[[1]])), 
                                                          rep(5,nrow(outDREAMzs_3p$chain[[1]])), 
                                                          rep(6,nrow(outDREAMzs_3p$chain[[1]])), 
                                                          rep(7,nrow(outDREAMzs_3p$chain[[1]])))))
outDREAMzs_3p$WAIC = waic(x = outDREAMzs_3p$pll)
outDREAMzs_3p$WAIC2 = waic2(x = outDREAMzs_3p$pll)
#Rejection rate
outDREAMzs_3p$rejectRate = rejectionRate(outDREAMzs_3p$chain)
#Highest Posterior Density Intervals
outDREAMzs_3p$HPD = HPDinterval(outDREAMzs_3p$chain)
#Credible Intervals
outDREAMzs_3p$CI = getCredibleIntervals(outDREAMzs_3p$chain[[1]])
#Effective sample size for each parameter should be similar
outDREAMzs_3p$Neff = effectiveSize(outDREAMzs_3p$chain[,1:3])

#Sample only the last 1000 from each chain
sample_chain_inds = (nrow(outDREAMzs_3p$chain[[1]])-1000):nrow(outDREAMzs_3p$chain[[1]])

png('DREAMzs3p/traceplot2.png', res = 300, units = 'in', width = 7, height = 7)
mcmc_trace(x = outDREAMzs_3p$chain[,1:3], size = 0.05)
dev.off()

png('DREAMzs3p/correlplot.png', res = 300, units = 'in', width = 7, height = 7)
correlationPlot(outDREAMzs_3p$chain[sample_chain_inds,1:3], method = 'spearman')
dev.off()

png('DREAMzs3p/gelmanplot.png', res = 300, units = 'in', width = 7, height = 7)
gelmanDiagnostics(outDREAMzs_3p, plot = TRUE, thin = 0)
dev.off()

png('DREAMzs3p/marginalplot.png', res = 300, units = 'in', width = 7, height = 7)
marginalPlot(x = outDREAMzs_3p$chain[sample_chain_inds,1:3], prior = prior3$sampler(n = 50000), singlePanel = FALSE, 
             xrange = t(rbind(c(-20,5),c(-10,5),c(-5,15))))
dev.off()

png('DREAMzs3p/PairPlot.png', res = 300, units = 'in', width = 7, height = 7)
mcmc_pairs(outDREAMzs_3p$chain[sample_chain_inds,1:3], diag_fun = 'dens', off_diag_fun = 'scatter', 
           off_diag_args = list(size=0.5,alpha=0.5), 
           xlim = rbind(c(-15,0),c(-6,1),c(0,6),c(-15,0),c(-6,1),c(0,6),c(-15,0),c(-6,1),c(0,6)),
           ylim = rbind(c(0,NA),c(-15,0),c(-15,0),c(-6,0),c(0,NA),c(-6,0),c(0,6),c(0,6),c(0,NA)))
dev.off()

png('DREAMzs3p/autocorrplot.png', res = 300, units = 'in', width = 7, height = 7)
autocorr.plot(x = outDREAMzs_3p$chain[[1]][-1,1:3], lag.max = 100)
dev.off()

png('DREAMzs3p/autocorrplot2.png', res = 300, units = 'in', width = 7, height = 7)
mcmc_acf(x = outDREAMzs_3p$chain, pars = outDREAMzs_3p$setup$names, lags = 20)
dev.off()

png('DREAMzs3p/densOverlay.png', res = 300, units = 'in', width = 7, height = 7)
mcmc_dens_overlay(outDREAMzs_3p$chain[sample_chain_inds,])
dev.off()

png('DREAMzs3p/HPD.png', res = 300, units = 'in', width = 7, height = 7)
mcmc_areas(x = outDREAMzs_3p$chain[sample_chain_inds,], pars = outDREAMzs_3p$setup$names, 
           point_est = 'mean', prob = 0.95)
dev.off()

png('DREAMzs3p/HPD_boxplot.png', res = 300, units = 'in', width = 7, height = 7)
mcmc_intervals(x = outDREAMzs_3p$chain[sample_chain_inds,], pars = outDREAMzs_3p$setup$names,
               prob_outer = 0.95)
dev.off()

geweke.plot(x = outDREAMzs_3p$chain[,1:3], ask = FALSE, nbins = 8)

stopParallel(setUpDREAMzs_3p)

# AGU Large Floods 1962-2020 FtSmith PCA----
#  DREAMzs: prior #2----
dir.create(path = 'DREAMzs_L15_SmithChipVermPrecipPCA_1962-2020', showWarnings = FALSE)
setUpDREAMzs_3p_AGU <- createBayesianSetup(likelihood_sm, prior = prior2, parallel = 7, 
                                       parallelOptions = list(packages=list('BayesianTools'), 
                                                              variables=list('X_holdsm','Y_holdsm', 'PCAcvs'), 
                                                              dlls=NULL), 
                                       names = c('Int', 'PC1', 'PC2'), 
                                       plotLower = c(-15, -1, -1), plotUpper = c(0, 6, 6), 
                                       plotBest = as.numeric(res_Firth_L15_PC3$coefficients))
set.seed(16421)
outDREAMzs_3p_AGU <- runMCMC(bayesianSetup = setUpDREAMzs_3p_AGU, sampler = "DREAMzs", settings = settingsDREAMzs_1)
#Remove all -Inf in chain. These result from poor initial conditions (prior samples)
outDREAMzs_3p_AGU = removeInf(outDREAMzs_3p_AGU)
#Check for remaining infinities
if(any(outDREAMzs_3p_AGU$InfCk > 0)){
  print('infinities remain within MCMC chain posteriors')
}
summary(outDREAMzs_3p_AGU)

#Compute WAIC and DIC
outDREAMzs_3p_AGU$WAICBT = WAIC(outDREAMzs_3p_AGU, numSamples = 100000)
outDREAMzs_3p_AGU$WAIC2BT = WAIC2(outDREAMzs_3p_AGU, numSamples = 100000)
outDREAMzs_3p_AGU$DIC = DIC(outDREAMzs_3p_AGU)
#Pointwise log-likelihoods
outDREAMzs_3p_AGU$pll = outDREAMzs_3p_AGU$setup$likelihood$density(getSample(outDREAMzs_3p_AGU, parametersOnly = F)[,1:outDREAMzs_3p_AGU$setup$numPars], sum = FALSE)
#LOO
outDREAMzs_3p_AGU$LOO = loo(outDREAMzs_3p_AGU$pll, cores = 7, save_psis = TRUE, 
                            r_eff = relative_eff(x = exp(outDREAMzs_3p_AGU$pll), 
                                                 chain_id = c(rep(1,nrow(outDREAMzs_3p_AGU$chain[[1]])), 
                                                              rep(2,nrow(outDREAMzs_3p_AGU$chain[[1]])), 
                                                              rep(3,nrow(outDREAMzs_3p_AGU$chain[[1]])), 
                                                              rep(4,nrow(outDREAMzs_3p_AGU$chain[[1]])), 
                                                              rep(5,nrow(outDREAMzs_3p_AGU$chain[[1]])), 
                                                              rep(6,nrow(outDREAMzs_3p_AGU$chain[[1]])), 
                                                              rep(7,nrow(outDREAMzs_3p_AGU$chain[[1]])))))
outDREAMzs_3p_AGU$WAIC = waic(x = outDREAMzs_3p_AGU$pll)
outDREAMzs_3p_AGU$WAIC2 = waic2(x = outDREAMzs_3p_AGU$pll)
#Rejection rate
outDREAMzs_3p_AGU$rejectRate = rejectionRate(outDREAMzs_3p_AGU$chain)
#Highest Posterior Density Intervals
outDREAMzs_3p_AGU$HPD = HPDinterval(outDREAMzs_3p_AGU$chain)
#Credible Intervals
outDREAMzs_3p_AGU$CI = getCredibleIntervals(outDREAMzs_3p_AGU$chain[[1]])
#Effective sample size for each parameter should be similar
outDREAMzs_3p_AGU$Neff = effectiveSize(outDREAMzs_3p_AGU$chain[,1:3])

#Sample only the last 1000 from each chain
sample_chain_inds = (nrow(outDREAMzs_3p_AGU$chain[[1]])-1000):nrow(outDREAMzs_3p_AGU$chain[[1]])

dir.create('DREAMzs_L15_SmithChipVermPrecipPCA_1962-2020/MCMC', showWarnings = FALSE)
png('DREAMzs_L15_SmithChipVermPrecipPCA_1962-2020/MCMC/traceplot2.png', res = 300, units = 'in', width = 7, height = 7)
mcmc_trace(x = outDREAMzs_3p_AGU$chain[,1:3], size = 0.05)
dev.off()

png('DREAMzs_L15_SmithChipVermPrecipPCA_1962-2020/MCMC/correlplot.png', res = 300, units = 'in', width = 7, height = 7)
correlationPlot(outDREAMzs_3p_AGU$chain[sample_chain_inds,1:3], method = 'spearman')
dev.off()

png('DREAMzs_L15_SmithChipVermPrecipPCA_1962-2020/MCMC/gelmanplot.png', res = 300, units = 'in', width = 7, height = 7)
gelmanDiagnostics(outDREAMzs_3p_AGU, plot = TRUE, thin = 0)
dev.off()

png('DREAMzs_L15_SmithChipVermPrecipPCA_1962-2020/MCMC/marginalplot.png', res = 300, units = 'in', width = 7, height = 7)
marginalPlot(x = outDREAMzs_3p_AGU$chain[sample_chain_inds,1:3], prior = prior2$sampler(n = 50000), singlePanel = FALSE, 
             xrange = t(rbind(c(-20,5),c(-10,5),c(-5,15))))
dev.off()

png('DREAMzs_L15_SmithChipVermPrecipPCA_1962-2020/MCMC/PairPlot.png', res = 300, units = 'in', width = 7, height = 7)
mcmc_pairs(outDREAMzs_3p_AGU$chain[sample_chain_inds,1:3], diag_fun = 'dens', off_diag_fun = 'scatter', 
           off_diag_args = list(size=0.5,alpha=0.5), 
           xlim = rbind(c(-15,0),c(-1,6),c(-1,6),c(-15,0),c(-1,6),c(-1,6),c(-15,0),c(-1,6),c(-1,6)),
           ylim = rbind(c(0,NA),c(-15,0),c(-15,0),c(-1,6),c(0,NA),c(-1,6),c(-1,6),c(-1,6),c(0,NA)))
dev.off()

png('DREAMzs_L15_SmithChipVermPrecipPCA_1962-2020/MCMC/autocorrplot.png', res = 300, units = 'in', width = 7, height = 7)
autocorr.plot(x = outDREAMzs_3p_AGU$chain[[1]][-1,1:3], lag.max = 100)
dev.off()

png('DREAMzs_L15_SmithChipVermPrecipPCA_1962-2020/MCMC/autocorrplot2.png', res = 300, units = 'in', width = 7, height = 7)
mcmc_acf(x = outDREAMzs_3p_AGU$chain, pars = outDREAMzs_3p_AGU$setup$names, lags = 20)
dev.off()

png('DREAMzs_L15_SmithChipVermPrecipPCA_1962-2020/MCMC/densOverlay.png', res = 300, units = 'in', width = 7, height = 7)
mcmc_dens_overlay(outDREAMzs_3p_AGU$chain[sample_chain_inds,])
dev.off()

png('DREAMzs_L15_SmithChipVermPrecipPCA_1962-2020/MCMC/HPD.png', res = 300, units = 'in', width = 7, height = 7)
mcmc_areas(x = outDREAMzs_3p_AGU$chain[sample_chain_inds,], pars = outDREAMzs_3p_AGU$setup$names, 
           point_est = 'mean', prob = 0.95)
dev.off()

png('DREAMzs_L15_SmithChipVermPrecipPCA_1962-2020/MCMC/HPD_boxplot.png', res = 300, units = 'in', width = 7, height = 7)
mcmc_intervals(x = outDREAMzs_3p_AGU$chain[sample_chain_inds,], pars = outDREAMzs_3p_AGU$setup$names,
               prob_outer = 0.95)
dev.off()

geweke.plot(x = outDREAMzs_3p_AGU$chain[,1:3], ask = FALSE, nbins = 8)

stopParallel(setUpDREAMzs_3p_AGU)

#Large Floods, 1915 - 2020, no data uncertainty----
#  DREAMzs: prior #2----
dir.create(path = 'DREAMzs_L15_SmithChipVermPrecipPCA_NoUncertainty_1915-2020', showWarnings = FALSE)
setUpDREAMzs_NoUncertainty <- createBayesianSetup(likelihood_sm_1915, prior = prior2, parallel = 7, 
                                           parallelOptions = list(packages=list('BayesianTools'), 
                                                                  variables=list('X_L15sm','Y_L15sm', 'PCAcvs'), 
                                                                  dlls=NULL), 
                                           names = c('Int', 'PC1', 'PC2'), 
                                           plotLower = c(-15, -1, -1), plotUpper = c(0, 6, 6), 
                                           plotBest = as.numeric(res_Firth_L15_PC3$coefficients))
set.seed(16421)
outDREAMzs_NoUncertainty <- runMCMC(bayesianSetup = setUpDREAMzs_NoUncertainty, sampler = "DREAMzs", 
                                    settings = settingsDREAMzs_1)
#Remove all -Inf in chain. These result from poor initial conditions (prior samples)
outDREAMzs_NoUncertainty = removeInf(outDREAMzs_NoUncertainty)
#Check for remaining infinities
if(any(outDREAMzs_NoUncertainty$InfCk > 0)){
  print('infinities remain within MCMC chain posteriors')
}
summary(outDREAMzs_NoUncertainty)

#Compute WAIC and DIC
outDREAMzs_NoUncertainty$WAICBT = WAIC(outDREAMzs_NoUncertainty, numSamples = 100000)
outDREAMzs_NoUncertainty$WAIC2BT = WAIC2(outDREAMzs_NoUncertainty, numSamples = 100000)
outDREAMzs_NoUncertainty$DIC = DIC(outDREAMzs_NoUncertainty)
#Pointwise log-likelihoods
outDREAMzs_NoUncertainty$pll = outDREAMzs_NoUncertainty$setup$likelihood$density(getSample(outDREAMzs_NoUncertainty, parametersOnly = F)[,1:outDREAMzs_NoUncertainty$setup$numPars], sum = FALSE)
#LOO
outDREAMzs_NoUncertainty$LOO = loo(outDREAMzs_NoUncertainty$pll, cores = 7, save_psis = TRUE, 
                            r_eff = relative_eff(x = exp(outDREAMzs_NoUncertainty$pll), 
                                                 chain_id = c(rep(1,nrow(outDREAMzs_NoUncertainty$chain[[1]])), 
                                                              rep(2,nrow(outDREAMzs_NoUncertainty$chain[[1]])), 
                                                              rep(3,nrow(outDREAMzs_NoUncertainty$chain[[1]])), 
                                                              rep(4,nrow(outDREAMzs_NoUncertainty$chain[[1]])), 
                                                              rep(5,nrow(outDREAMzs_NoUncertainty$chain[[1]])), 
                                                              rep(6,nrow(outDREAMzs_NoUncertainty$chain[[1]])), 
                                                              rep(7,nrow(outDREAMzs_NoUncertainty$chain[[1]])))))
outDREAMzs_NoUncertainty$WAIC = waic(x = outDREAMzs_NoUncertainty$pll)
outDREAMzs_NoUncertainty$WAIC2 = waic2(x = outDREAMzs_NoUncertainty$pll)
#Rejection rate
outDREAMzs_NoUncertainty$rejectRate = rejectionRate(outDREAMzs_NoUncertainty$chain)
#Highest Posterior Density Intervals
outDREAMzs_NoUncertainty$HPD = HPDinterval(outDREAMzs_NoUncertainty$chain)
#Credible Intervals
outDREAMzs_NoUncertainty$CI = getCredibleIntervals(outDREAMzs_NoUncertainty$chain[[1]])
#Effective sample size for each parameter should be similar
outDREAMzs_NoUncertainty$Neff = effectiveSize(outDREAMzs_NoUncertainty$chain[,1:3])

#Sample only the last 1000 from each chain
sample_chain_inds = (nrow(outDREAMzs_NoUncertainty$chain[[1]])-1000):nrow(outDREAMzs_NoUncertainty$chain[[1]])

dir.create('DREAMzs_L15_SmithChipVermPrecipPCA_NoUncertainty_1915-2020/MCMC', showWarnings = FALSE)
png('DREAMzs_L15_SmithChipVermPrecipPCA_NoUncertainty_1915-2020/MCMC/traceplot2.png', res = 300, units = 'in', width = 7, height = 7)
mcmc_trace(x = outDREAMzs_NoUncertainty$chain[,1:3], size = 0.05)
dev.off()

png('DREAMzs_L15_SmithChipVermPrecipPCA_NoUncertainty_1915-2020/MCMC/correlplot.png', res = 300, units = 'in', width = 7, height = 7)
correlationPlot(outDREAMzs_NoUncertainty$chain[sample_chain_inds,1:3], method = 'spearman')
dev.off()

png('DREAMzs_L15_SmithChipVermPrecipPCA_NoUncertainty_1915-2020/MCMC/gelmanplot.png', res = 300, units = 'in', width = 7, height = 7)
gelmanDiagnostics(outDREAMzs_NoUncertainty, plot = TRUE, thin = 0)
dev.off()

png('DREAMzs_L15_SmithChipVermPrecipPCA_NoUncertainty_1915-2020/MCMC/marginalplot.png', res = 300, units = 'in', width = 7, height = 7)
marginalPlot(x = outDREAMzs_NoUncertainty$chain[sample_chain_inds,1:3], prior = prior2$sampler(n = 50000), singlePanel = FALSE, 
             xrange = t(rbind(c(-20,5),c(-10,5),c(-5,15))))
dev.off()

png('DREAMzs_L15_SmithChipVermPrecipPCA_NoUncertainty_1915-2020/MCMC/PairPlot.png', res = 300, units = 'in', width = 7, height = 7)
mcmc_pairs(outDREAMzs_NoUncertainty$chain[sample_chain_inds,1:3], diag_fun = 'dens', off_diag_fun = 'scatter', 
           off_diag_args = list(size=0.5,alpha=0.5), 
           xlim = rbind(c(-15,0),c(-1,6),c(-1,6),c(-15,0),c(-1,6),c(-1,6),c(-15,0),c(-1,6),c(-1,6)),
           ylim = rbind(c(0,NA),c(-15,0),c(-15,0),c(-1,6),c(0,NA),c(-1,6),c(-1,6),c(-1,6),c(0,NA)))
dev.off()

png('DREAMzs_L15_SmithChipVermPrecipPCA_NoUncertainty_1915-2020/MCMC/autocorrplot.png', res = 300, units = 'in', width = 7, height = 7)
autocorr.plot(x = outDREAMzs_NoUncertainty$chain[[1]][-1,1:3], lag.max = 100)
dev.off()

png('DREAMzs_L15_SmithChipVermPrecipPCA_NoUncertainty_1915-2020/MCMC/autocorrplot2.png', res = 300, units = 'in', width = 7, height = 7)
mcmc_acf(x = outDREAMzs_NoUncertainty$chain, pars = outDREAMzs_NoUncertainty$setup$names, lags = 20)
dev.off()

png('DREAMzs_L15_SmithChipVermPrecipPCA_NoUncertainty_1915-2020/MCMC/densOverlay.png', res = 300, units = 'in', width = 7, height = 7)
mcmc_dens_overlay(outDREAMzs_NoUncertainty$chain[sample_chain_inds,])
dev.off()

png('DREAMzs_L15_SmithChipVermPrecipPCA_NoUncertainty_1915-2020/MCMC/HPD.png', res = 300, units = 'in', width = 7, height = 7)
mcmc_areas(x = outDREAMzs_NoUncertainty$chain[sample_chain_inds,], pars = outDREAMzs_NoUncertainty$setup$names, 
           point_est = 'mean', prob = 0.95)
dev.off()

png('DREAMzs_L15_SmithChipVermPrecipPCA_NoUncertainty_1915-2020/MCMC/HPD_boxplot.png', res = 300, units = 'in', width = 7, height = 7)
mcmc_intervals(x = outDREAMzs_NoUncertainty$chain[sample_chain_inds,], pars = outDREAMzs_NoUncertainty$setup$names,
               prob_outer = 0.95)
dev.off()

geweke.plot(x = outDREAMzs_NoUncertainty$chain[,1:3], ask = FALSE, nbins = 8)

stopParallel(setUpDREAMzs_NoUncertainty)

#Large Floods, 1915 - 2020----
# Prior Definition----
#Parameters are from Firth regression with all large floods
#Followed by eta for historical large
#Followed by eta for historical moderate
#Followed by eta for historical small
#Followed by eta for years with unknown magnitude
#Followed by theta for historical no flood
#Uniform priors
density_L15_unif_SensSpec_Hist = function(par){
  d1 = dnorm(par[1], mean = as.numeric(res_Firth_L15_PC3$coefficients)[1], sd = 10, log =TRUE)
  d2 = dnorm(par[2], mean = as.numeric(res_Firth_L15_PC3$coefficients)[2], sd = 10, log =TRUE)
  d3 = dnorm(par[3], mean = as.numeric(res_Firth_L15_PC3$coefficients)[3], sd = 10, log =TRUE)
  d4 = dbeta(par[4], shape1 = 1, shape2 = 1, log =TRUE)
  d5 = dbeta(par[5], shape1 = 1, shape2 = 1, log =TRUE)
  d6 = dbeta(par[6], shape1 = 1, shape2 = 1, log =TRUE)
  d7 = dbeta(par[7], shape1 = 1, shape2 = 1, log =TRUE)
  d8 = dbeta(par[8], shape1 = 1, shape2 = 1, log =TRUE)
  return(d1 + d2 + d3 + d4 + d5 + d6 + d7 + d8)
}
sampler_L15_unif_SensSpec_Hist = function(n=1){
  d1 = rnorm(n, mean = as.numeric(res_Firth_L15_PC3$coefficients)[1], sd = 10)
  d2 = rnorm(n, mean = as.numeric(res_Firth_L15_PC3$coefficients)[2], sd = 10)
  d3 = rnorm(n, mean = as.numeric(res_Firth_L15_PC3$coefficients)[3], sd = 10)
  d4 = rbeta(n, shape1 = 1, shape2 = 1)
  d5 = rbeta(n, shape1 = 1, shape2 = 1)
  d6 = rbeta(n, shape1 = 1, shape2 = 1)
  d7 = rbeta(n, shape1 = 1, shape2 = 1)
  d8 = rbeta(n, shape1 = 1, shape2 = 1)
  return(cbind(d1,d2,d3,d4,d5,d6,d7,d8))
}
prior_L15_unif_SensSpec_Hist = createPrior(density = density_L15_unif_SensSpec_Hist, 
                                           sampler = sampler_L15_unif_SensSpec_Hist)

#Parameters are from Firth regression with all large floods
#Followed by eta for all historical large
#Followed by eta for all historical moderate
#Followed by eta for all historical small
#Followed by eta for all years with unknown magnitude
#Followed by theta for historical no flood
#Uniform priors
density_L15_unif_SensSpec_pAll = function(par){
  d1 = dnorm(par[1], mean = as.numeric(res_Firth_L15_PC3$coefficients)[1], sd = 10, log =TRUE)
  d2 = dnorm(par[2], mean = as.numeric(res_Firth_L15_PC3$coefficients)[2], sd = 10, log =TRUE)
  d3 = dnorm(par[3], mean = as.numeric(res_Firth_L15_PC3$coefficients)[3], sd = 10, log =TRUE)
  #Large
  d4 = dbeta(par[4], shape1 = 1, shape2 = 1, log =TRUE)
  d5 = dbeta(par[5], shape1 = 1, shape2 = 1, log =TRUE)
  d6 = dbeta(par[6], shape1 = 1, shape2 = 1, log =TRUE)
  d7 = dbeta(par[7], shape1 = 1, shape2 = 1, log =TRUE)
  d8 = dbeta(par[8], shape1 = 1, shape2 = 1, log =TRUE)
  d9 = dbeta(par[9], shape1 = 1, shape2 = 1, log =TRUE)
  #Moderate
  d10 = dbeta(par[10], shape1 = 1, shape2 = 1, log =TRUE)
  d11 = dbeta(par[11], shape1 = 1, shape2 = 1, log =TRUE)
  d12 = dbeta(par[12], shape1 = 1, shape2 = 1, log =TRUE)
  d13 = dbeta(par[13], shape1 = 1, shape2 = 1, log =TRUE)
  d14 = dbeta(par[14], shape1 = 1, shape2 = 1, log =TRUE)
  #Small
  d15 = dbeta(par[15], shape1 = 1, shape2 = 1, log =TRUE)
  d16 = dbeta(par[16], shape1 = 1, shape2 = 1, log =TRUE)
  d17 = dbeta(par[17], shape1 = 1, shape2 = 1, log =TRUE)
  d18 = dbeta(par[18], shape1 = 1, shape2 = 1, log =TRUE)
  d19 = dbeta(par[19], shape1 = 1, shape2 = 1, log =TRUE)
  #Unknown
  d20 = dbeta(par[20], shape1 = 1, shape2 = 1, log =TRUE)
  d21 = dbeta(par[21], shape1 = 1, shape2 = 1, log =TRUE)
  d22 = dbeta(par[22], shape1 = 1, shape2 = 1, log =TRUE)
  d23 = dbeta(par[23], shape1 = 1, shape2 = 1, log =TRUE)
  #Theta
  d24 = dbeta(par[24], shape1 = 1, shape2 = 1, log =TRUE)
  return(d1 + d2 + d3 + d4 + d5 + d6 + d7 + d8 + d9 + d10 + d11 + d12 + d13 + d14 + d15 + d16 + d17 + d18 + d19 + d20 + d21 + d22 + d23 + d24)
}
sampler_L15_unif_SensSpec_pAll = function(n=1){
  d1 = rnorm(n, mean = as.numeric(res_Firth_L15_PC3$coefficients)[1], sd = 10)
  d2 = rnorm(n, mean = as.numeric(res_Firth_L15_PC3$coefficients)[2], sd = 10)
  d3 = rnorm(n, mean = as.numeric(res_Firth_L15_PC3$coefficients)[3], sd = 10)
  d4 = rbeta(n, shape1 = 1, shape2 = 1)
  d5 = rbeta(n, shape1 = 1, shape2 = 1)
  d6 = rbeta(n, shape1 = 1, shape2 = 1)
  d7 = rbeta(n, shape1 = 1, shape2 = 1)
  d8 = rbeta(n, shape1 = 1, shape2 = 1)
  d9 = rbeta(n, shape1 = 1, shape2 = 1)
  d10 = rbeta(n, shape1 = 1, shape2 = 1)
  d11 = rbeta(n, shape1 = 1, shape2 = 1)
  d12 = rbeta(n, shape1 = 1, shape2 = 1)
  d13 = rbeta(n, shape1 = 1, shape2 = 1)
  d14 = rbeta(n, shape1 = 1, shape2 = 1)
  d15 = rbeta(n, shape1 = 1, shape2 = 1)
  d16 = rbeta(n, shape1 = 1, shape2 = 1)
  d17 = rbeta(n, shape1 = 1, shape2 = 1)
  d18 = rbeta(n, shape1 = 1, shape2 = 1)
  d19 = rbeta(n, shape1 = 1, shape2 = 1)
  d20 = rbeta(n, shape1 = 1, shape2 = 1)
  d21 = rbeta(n, shape1 = 1, shape2 = 1)
  d22 = rbeta(n, shape1 = 1, shape2 = 1)
  d23 = rbeta(n, shape1 = 1, shape2 = 1)
  d24 = rbeta(n, shape1 = 1, shape2 = 1)
  return(cbind(d1,d2,d3,d4,d5,d6,d7,d8,d9,d10,d11,d12,d13,d14,d15,d16,d17,d18,d19,d20,d21,d22,d23,d24))
}
prior_L15_unif_SensSpec_pAll = createPrior(density = density_L15_unif_SensSpec_pAll, 
                                           sampler = sampler_L15_unif_SensSpec_pAll)


density_L15_unif_SensSpec_pAll_theta = function(par){
  d1 = dnorm(par[1], mean = as.numeric(res_Firth_L15_PC3$coefficients)[1], sd = 10, log =TRUE)
  d2 = dnorm(par[2], mean = as.numeric(res_Firth_L15_PC3$coefficients)[2], sd = 10, log =TRUE)
  d3 = dnorm(par[3], mean = as.numeric(res_Firth_L15_PC3$coefficients)[3], sd = 10, log =TRUE)
  #Large
  d4 = dbeta(par[4], shape1 = 1, shape2 = 1, log =TRUE)
  d5 = dbeta(par[5], shape1 = 1, shape2 = 1, log =TRUE)
  d6 = dbeta(par[6], shape1 = 1, shape2 = 1, log =TRUE)
  d7 = dbeta(par[7], shape1 = 1, shape2 = 1, log =TRUE)
  d8 = dbeta(par[8], shape1 = 1, shape2 = 1, log =TRUE)
  d9 = dbeta(par[9], shape1 = 1, shape2 = 1, log =TRUE)
  #Moderate
  d10 = dbeta(par[10], shape1 = 1, shape2 = 1, log =TRUE)
  d11 = dbeta(par[11], shape1 = 1, shape2 = 1, log =TRUE)
  d12 = dbeta(par[12], shape1 = 1, shape2 = 1, log =TRUE)
  d13 = dbeta(par[13], shape1 = 1, shape2 = 1, log =TRUE)
  d14 = dbeta(par[14], shape1 = 1, shape2 = 1, log =TRUE)
  #Small
  d15 = dbeta(par[15], shape1 = 1, shape2 = 1, log =TRUE)
  d16 = dbeta(par[16], shape1 = 1, shape2 = 1, log =TRUE)
  d17 = dbeta(par[17], shape1 = 1, shape2 = 1, log =TRUE)
  d18 = dbeta(par[18], shape1 = 1, shape2 = 1, log =TRUE)
  d19 = dbeta(par[19], shape1 = 1, shape2 = 1, log =TRUE)
  #Unknown
  d20 = dbeta(par[20], shape1 = 1, shape2 = 1, log =TRUE)
  d21 = dbeta(par[21], shape1 = 1, shape2 = 1, log =TRUE)
  d22 = dbeta(par[22], shape1 = 1, shape2 = 1, log =TRUE)
  d23 = dbeta(par[23], shape1 = 1, shape2 = 1, log =TRUE)
  #Theta
  d24 = dbeta(par[24], shape1 = 1, shape2 = 1, log =TRUE)
  d25 = dbeta(par[25], shape1 = 1, shape2 = 1, log =TRUE)
  d26 = dbeta(par[26], shape1 = 1, shape2 = 1, log =TRUE)
  d27 = dbeta(par[27], shape1 = 1, shape2 = 1, log =TRUE)
  d28 = dbeta(par[28], shape1 = 1, shape2 = 1, log =TRUE)
  d29 = dbeta(par[29], shape1 = 1, shape2 = 1, log =TRUE)
  d30 = dbeta(par[30], shape1 = 1, shape2 = 1, log =TRUE)
  d31 = dbeta(par[31], shape1 = 1, shape2 = 1, log =TRUE)
  d32 = dbeta(par[32], shape1 = 1, shape2 = 1, log =TRUE)
  d33 = dbeta(par[33], shape1 = 1, shape2 = 1, log =TRUE)
  d34 = dbeta(par[34], shape1 = 1, shape2 = 1, log =TRUE)
  d35 = dbeta(par[35], shape1 = 1, shape2 = 1, log =TRUE)
  d36 = dbeta(par[36], shape1 = 1, shape2 = 1, log =TRUE)
  d37 = dbeta(par[37], shape1 = 1, shape2 = 1, log =TRUE)
  d38 = dbeta(par[38], shape1 = 1, shape2 = 1, log =TRUE)
  d39 = dbeta(par[39], shape1 = 1, shape2 = 1, log =TRUE)
  d40 = dbeta(par[40], shape1 = 1, shape2 = 1, log =TRUE)
  d41 = dbeta(par[41], shape1 = 1, shape2 = 1, log =TRUE)
  d42 = dbeta(par[42], shape1 = 1, shape2 = 1, log =TRUE)
  d43 = dbeta(par[43], shape1 = 1, shape2 = 1, log =TRUE)
  d44 = dbeta(par[44], shape1 = 1, shape2 = 1, log =TRUE)
  d45 = dbeta(par[45], shape1 = 1, shape2 = 1, log =TRUE)
  d46 = dbeta(par[46], shape1 = 1, shape2 = 1, log =TRUE)
  return(d1 + d2 + d3 + d4 + d5 + d6 + d7 + d8 + d9 + d10 + d11 + d12 + 
           d13 + d14 + d15 + d16 + d17 + d18 + d19 + d20 + d21 + d22 + 
           d23 + d24 + d25 + d26 + d27 + d28 + d29 + d30 + d31 + d32 +
           d33 + d34 + d35 + d36 + d37 + d38 + d39 + d40 + d41 + d42 +
           d43 + d44 + d45 + d46)
}
sampler_L15_unif_SensSpec_pAll_theta = function(n=1){
  d1 = rnorm(n, mean = as.numeric(res_Firth_L15_PC3$coefficients)[1], sd = 10)
  d2 = rnorm(n, mean = as.numeric(res_Firth_L15_PC3$coefficients)[2], sd = 10)
  d3 = rnorm(n, mean = as.numeric(res_Firth_L15_PC3$coefficients)[3], sd = 10)
  d4 = rbeta(n, shape1 = 1, shape2 = 1)
  d5 = rbeta(n, shape1 = 1, shape2 = 1)
  d6 = rbeta(n, shape1 = 1, shape2 = 1)
  d7 = rbeta(n, shape1 = 1, shape2 = 1)
  d8 = rbeta(n, shape1 = 1, shape2 = 1)
  d9 = rbeta(n, shape1 = 1, shape2 = 1)
  d10 = rbeta(n, shape1 = 1, shape2 = 1)
  d11 = rbeta(n, shape1 = 1, shape2 = 1)
  d12 = rbeta(n, shape1 = 1, shape2 = 1)
  d13 = rbeta(n, shape1 = 1, shape2 = 1)
  d14 = rbeta(n, shape1 = 1, shape2 = 1)
  d15 = rbeta(n, shape1 = 1, shape2 = 1)
  d16 = rbeta(n, shape1 = 1, shape2 = 1)
  d17 = rbeta(n, shape1 = 1, shape2 = 1)
  d18 = rbeta(n, shape1 = 1, shape2 = 1)
  d19 = rbeta(n, shape1 = 1, shape2 = 1)
  d20 = rbeta(n, shape1 = 1, shape2 = 1)
  d21 = rbeta(n, shape1 = 1, shape2 = 1)
  d22 = rbeta(n, shape1 = 1, shape2 = 1)
  d23 = rbeta(n, shape1 = 1, shape2 = 1)
  d24 = rbeta(n, shape1 = 1, shape2 = 1)
  d25 = rbeta(n, shape1 = 1, shape2 = 1)
  d26 = rbeta(n, shape1 = 1, shape2 = 1)
  d27 = rbeta(n, shape1 = 1, shape2 = 1)
  d28 = rbeta(n, shape1 = 1, shape2 = 1)
  d29 = rbeta(n, shape1 = 1, shape2 = 1)
  d30 = rbeta(n, shape1 = 1, shape2 = 1)
  d31 = rbeta(n, shape1 = 1, shape2 = 1)
  d32 = rbeta(n, shape1 = 1, shape2 = 1)
  d33 = rbeta(n, shape1 = 1, shape2 = 1)
  d34 = rbeta(n, shape1 = 1, shape2 = 1)
  d35 = rbeta(n, shape1 = 1, shape2 = 1)
  d36 = rbeta(n, shape1 = 1, shape2 = 1)
  d37 = rbeta(n, shape1 = 1, shape2 = 1)
  d38 = rbeta(n, shape1 = 1, shape2 = 1)
  d39 = rbeta(n, shape1 = 1, shape2 = 1)
  d40 = rbeta(n, shape1 = 1, shape2 = 1)
  d41 = rbeta(n, shape1 = 1, shape2 = 1)
  d42 = rbeta(n, shape1 = 1, shape2 = 1)
  d43 = rbeta(n, shape1 = 1, shape2 = 1)
  d44 = rbeta(n, shape1 = 1, shape2 = 1)
  d45 = rbeta(n, shape1 = 1, shape2 = 1)
  d46 = rbeta(n, shape1 = 1, shape2 = 1)
  return(cbind(d1,d2,d3,d4,d5,d6,d7,d8,d9,d10,d11,d12,
               d13,d14,d15,d16,d17,d18,d19,d20,d21,d22,
               d23,d24,d25,d26,d27,d28,d29,d30,d31,d32,
               d33,d34,d35,d36,d37,d38,d39,d40,d41,d42,
               d43,d44,d45,d46))
}
prior_L15_unif_SensSpec_pAll_theta = createPrior(density = density_L15_unif_SensSpec_pAll_theta, 
                                                 sampler = sampler_L15_unif_SensSpec_pAll_theta)

# Likelihood Definition----
likelihood_L15_FixY_pSensSpec_PCAcvs_Historical1s <- function(param, sum = TRUE){
  #Determine indices of historical large Y = 1
  IndL = which((FloodMag_L15sm == 'L') & (years_L15sm < 1962))
  #Determine indices of historical moderate Y = 1
  IndM = which((FloodMag_L15sm == 'M') & (years_L15sm < 1962))
  #Determine indices of historical small Y = 1
  IndS = which((FloodMag_L15sm == 'S') & (years_L15sm < 1962))
  #Determine indices of historical not a flood Y = 0
  IndN = which((FloodMag_L15sm == 'N') & (years_L15sm < 1962))
  #Same for floods of unknown magnitude Y = 1
  IndU = which((FloodMag_L15sm == 'U'))
  
  #point likelihoods
  Xi = as.matrix(cbind(X_L15sm[,1], predPCAcvs[,c(1,2)]))
  #numerator
  num = exp(param[1] + param[2]*Xi[,2] + param[3]*Xi[,3])
  #y-hat
  pred = num/(1+num)
  #weights - assuming pred instead of q
  w = diag(pred * (1-pred))
  #Information matrix
  I = t(Xi) %*% w %*% Xi
  
  #Terms for sens spec likelihood
  pred10 = pred
  pred11 = pred
  pred00 = pred
  #Large - correct by eta
  pred11[IndL] = pred[IndL]*param[4]
  #Moderate - correct by eta
  pred11[IndM] = pred[IndM]*param[5]
  #Small - correct by eta
  pred11[IndS] = pred[IndS]*param[6]
  #Unknown - correct by eta
  pred11[IndU] = pred[IndU]*param[7]
  #Not a flood - correct by theta
  pred00[IndN] = (1-pred[IndN])*param[8]
  #All historical Y = 1 are affected by 1-theta
  pred10 = (1-pred)*(1-param[8])
  
  pll = vector('numeric', length=length(Y_L15sm))
  #Weighted 1-eta
  s01 = ((1-param[5])*length(IndM) + (1-param[6])*length(IndS) + (1-param[7])*length(IndU) + (1-param[4])*length(IndL))/(length(IndM) + length(IndS) + length(IndU) + length(IndL))
  for (j in 1:length(Y_L15sm)){
    if (j %in% IndL){
      #Get adjusted probability of Y = 1
      pll[j] = log(pred11[j] + pred10[j])
    }else if (j %in% IndM){
      #Get adjusted probability of Y = 1
      pll[j] = log(pred11[j] + pred10[j])
    }else if (j %in% IndS){
      #Get adjusted probability of Y = 1
      pll[j] = log(pred11[j] + pred10[j])
    }else if (j %in% IndU){
      #Get adjusted probability of Y = 1
      pll[j] = log(pred11[j] + pred10[j])
    }else if (j %in% IndN){
      #Get adjusted probability of Y = 0
      pll[j] = log(pred[j]*s01 + pred00[j])
    }else{
      #Could be Y=0 or Y=1 and Y is not uncertain
      pll[j] = Y_L15sm[j]*log(pred[j]) + (1 - Y_L15sm[j])*log(1 - pred[j])
    }
  }
  #sum with Firth correction
  ll = sum(pll) + 0.5*determinant(I, logarithm = TRUE)$modulus[1]
  if (is.nan(ll)){
    ll = -Inf
  }
  
  #Return sum of the log-likelihoods
  return(if (sum == TRUE) ll else (pll + 0.5*determinant(I, logarithm = TRUE)$modulus[1]/length(Y_L15sm)))  
}

likelihood_L15_FixY_pSensSpec_PCAcvs_Historical1s_pAll <- function(param, sum = TRUE){
  #Determine indices of historical large Y = 1
  IndL = which((FloodMag_L15sm == 'L') & (years_L15sm < 1962))
  #Determine indices of historical moderate Y = 1
  IndM = which((FloodMag_L15sm == 'M') & (years_L15sm < 1962))
  #Determine indices of historical small Y = 1
  IndS = which((FloodMag_L15sm == 'S') & (years_L15sm < 1962))
  #Determine indices of historical not a flood Y = 0
  IndN = which((FloodMag_L15sm == 'N') & (years_L15sm < 1962))
  #Same for floods of unknown magnitude Y = 1
  IndU = which((FloodMag_L15sm == 'U'))
  
  #point likelihoods
  Xi = as.matrix(cbind(X_L15sm[,1], predPCAcvs[,c(1,2)]))
  #numerator
  num = exp(param[1] + param[2]*Xi[,2] + param[3]*Xi[,3])
  #y-hat
  pred = num/(1+num)
  #weights - assuming pred instead of q
  w = diag(pred * (1-pred))
  #Information matrix
  I = t(Xi) %*% w %*% Xi
  
  #Terms for sens spec likelihood
  pred10 = pred
  pred11 = pred
  pred00 = pred
  #Large - correct by eta
  pred11[IndL[1]] = pred[IndL[1]]*param[4]
  pred11[IndL[2]] = pred[IndL[2]]*param[5]
  pred11[IndL[3]] = pred[IndL[3]]*param[6]
  pred11[IndL[4]] = pred[IndL[4]]*param[7]
  pred11[IndL[5]] = pred[IndL[5]]*param[8]
  pred11[IndL[6]] = pred[IndL[6]]*param[9]
  #Moderate - correct by eta
  pred11[IndM[1]] = pred[IndM[1]]*param[10]
  pred11[IndM[2]] = pred[IndM[2]]*param[11]
  pred11[IndM[3]] = pred[IndM[3]]*param[12]
  pred11[IndM[4]] = pred[IndM[4]]*param[13]
  pred11[IndM[5]] = pred[IndM[5]]*param[14]
  #Small - correct by eta
  pred11[IndS[1]] = pred[IndS[1]]*param[15]
  pred11[IndS[2]] = pred[IndS[2]]*param[16]
  pred11[IndS[3]] = pred[IndS[3]]*param[17]
  pred11[IndS[4]] = pred[IndS[4]]*param[18]
  pred11[IndS[5]] = pred[IndS[5]]*param[19]
  #Unknown - correct by eta
  pred11[IndU[1]] = pred[IndU[1]]*param[20]
  pred11[IndU[2]] = pred[IndU[2]]*param[21]
  pred11[IndU[3]] = pred[IndU[3]]*param[22]
  pred11[IndU[4]] = pred[IndU[4]]*param[23]
  #Not a flood - correct by theta
  pred00[IndN] = (1-pred[IndN])*param[24]
  #All historical Y = 1 are affected by 1-theta
  pred10 = (1-pred)*(1-param[24])
  
  pll = vector('numeric', length=length(Y_L15sm))
  #Weighted 1-eta
  s01 = sum(1-param[4:23])/(length(param[4:23]))
  for (j in 1:length(Y_L15sm)){
    if (j %in% IndL){
      #Get adjusted probability of Y = 1
      pll[j] = log(pred11[j] + pred10[j])
    }else if (j %in% IndM){
      #Get adjusted probability of Y = 1
      pll[j] = log(pred11[j] + pred10[j])
    }else if (j %in% IndS){
      #Get adjusted probability of Y = 1
      pll[j] = log(pred11[j] + pred10[j])
    }else if (j %in% IndU){
      #Get adjusted probability of Y = 1
      pll[j] = log(pred11[j] + pred10[j])
    }else if (j %in% IndN){
      #Get adjusted probability of Y = 0
      pll[j] = log(pred[j]*s01 + pred00[j])
    }else{
      #Could be Y=0 or Y=1 and Y is not uncertain
      pll[j] = Y_L15sm[j]*log(pred[j]) + (1 - Y_L15sm[j])*log(1 - pred[j])
    }
  }
  #sum with Firth correction
  ll = sum(pll) + 0.5*determinant(I, logarithm = TRUE)$modulus[1]
  if (is.nan(ll)){
    ll = -Inf
  }
  
  #Return sum of the log-likelihoods
  return(if (sum == TRUE) ll else (pll + 0.5*determinant(I, logarithm = TRUE)$modulus[1]/length(Y_L15sm)))  
}

likelihood_L15_FixY_pSensSpec_PCAcvs_Historical1s_pAll_theta <- function(param, sum = TRUE){
  #Determine indices of historical large Y = 1
  IndL = which((FloodMag_L15sm == 'L') & (years_L15sm < 1962))
  #Determine indices of historical moderate Y = 1
  IndM = which((FloodMag_L15sm == 'M') & (years_L15sm < 1962))
  #Determine indices of historical small Y = 1
  IndS = which((FloodMag_L15sm == 'S') & (years_L15sm < 1962))
  #Determine indices of historical not a flood Y = 0
  IndN = which((FloodMag_L15sm == 'N') & (years_L15sm < 1962))
  #Same for floods of unknown magnitude Y = 1
  IndU = which((FloodMag_L15sm == 'U'))
  
  #point likelihoods
  Xi = as.matrix(cbind(X_L15sm[,1], predPCAcvs[,c(1,2)]))
  #numerator
  num = exp(param[1] + param[2]*Xi[,2] + param[3]*Xi[,3])
  #y-hat
  pred = num/(1+num)
  #weights - assuming pred instead of q
  w = diag(pred * (1-pred))
  #Information matrix
  I = t(Xi) %*% w %*% Xi
  
  #Terms for sens spec likelihood
  pred10 = pred
  pred11 = pred
  pred00 = pred
  #Large - correct by eta
  pred11[IndL[1]] = pred[IndL[1]]*param[4]
  pred11[IndL[2]] = pred[IndL[2]]*param[5]
  pred11[IndL[3]] = pred[IndL[3]]*param[6]
  pred11[IndL[4]] = pred[IndL[4]]*param[7]
  pred11[IndL[5]] = pred[IndL[5]]*param[8]
  pred11[IndL[6]] = pred[IndL[6]]*param[9]
  #Moderate - correct by eta
  pred11[IndM[1]] = pred[IndM[1]]*param[10]
  pred11[IndM[2]] = pred[IndM[2]]*param[11]
  pred11[IndM[3]] = pred[IndM[3]]*param[12]
  pred11[IndM[4]] = pred[IndM[4]]*param[13]
  pred11[IndM[5]] = pred[IndM[5]]*param[14]
  #Small - correct by eta
  pred11[IndS[1]] = pred[IndS[1]]*param[15]
  pred11[IndS[2]] = pred[IndS[2]]*param[16]
  pred11[IndS[3]] = pred[IndS[3]]*param[17]
  pred11[IndS[4]] = pred[IndS[4]]*param[18]
  pred11[IndS[5]] = pred[IndS[5]]*param[19]
  #Unknown - correct by eta
  pred11[IndU[1]] = pred[IndU[1]]*param[20]
  pred11[IndU[2]] = pred[IndU[2]]*param[21]
  pred11[IndU[3]] = pred[IndU[3]]*param[22]
  pred11[IndU[4]] = pred[IndU[4]]*param[23]
  #Not a flood - correct by theta
  pred00[IndN[1]] = (1-pred[IndN[1]])*param[24]
  pred00[IndN[2]] = (1-pred[IndN[2]])*param[25]
  pred00[IndN[3]] = (1-pred[IndN[3]])*param[26]
  pred00[IndN[4]] = (1-pred[IndN[4]])*param[27]
  pred00[IndN[5]] = (1-pred[IndN[5]])*param[28]
  pred00[IndN[6]] = (1-pred[IndN[6]])*param[29]
  pred00[IndN[7]] = (1-pred[IndN[7]])*param[30]
  pred00[IndN[8]] = (1-pred[IndN[8]])*param[31]
  pred00[IndN[9]] = (1-pred[IndN[9]])*param[32]
  pred00[IndN[10]] = (1-pred[IndN[10]])*param[33]
  pred00[IndN[11]] = (1-pred[IndN[11]])*param[34]
  pred00[IndN[12]] = (1-pred[IndN[12]])*param[35]
  pred00[IndN[13]] = (1-pred[IndN[13]])*param[36]
  pred00[IndN[14]] = (1-pred[IndN[14]])*param[37]
  pred00[IndN[15]] = (1-pred[IndN[15]])*param[38]
  pred00[IndN[16]] = (1-pred[IndN[16]])*param[39]
  pred00[IndN[17]] = (1-pred[IndN[17]])*param[40]
  pred00[IndN[18]] = (1-pred[IndN[18]])*param[41]
  pred00[IndN[19]] = (1-pred[IndN[19]])*param[42]
  pred00[IndN[20]] = (1-pred[IndN[20]])*param[43]
  pred00[IndN[21]] = (1-pred[IndN[21]])*param[44]
  pred00[IndN[22]] = (1-pred[IndN[22]])*param[45]
  pred00[IndN[23]] = (1-pred[IndN[23]])*param[46]
  #All historical Y = 1 are affected by 1-theta
  #Weighted 1-theta
  s10 = sum(1-param[24:46])/(length(param[24:46]))
  pred10 = (1-pred)*s10
  
  pll = vector('numeric', length=length(Y_L15sm))
  #Weighted 1-eta
  s01 = sum(1-param[4:23])/(length(param[4:23]))
  for (j in 1:length(Y_L15sm)){
    if (j %in% IndL){
      #Get adjusted probability of Y = 1
      pll[j] = log(pred11[j] + pred10[j])
    }else if (j %in% IndM){
      #Get adjusted probability of Y = 1
      pll[j] = log(pred11[j] + pred10[j])
    }else if (j %in% IndS){
      #Get adjusted probability of Y = 1
      pll[j] = log(pred11[j] + pred10[j])
    }else if (j %in% IndU){
      #Get adjusted probability of Y = 1
      pll[j] = log(pred11[j] + pred10[j])
    }else if (j %in% IndN){
      #Get adjusted probability of Y = 0
      pll[j] = log(pred[j]*s01 + pred00[j])
    }else{
      #Could be Y=0 or Y=1 and Y is not uncertain
      pll[j] = Y_L15sm[j]*log(pred[j]) + (1 - Y_L15sm[j])*log(1 - pred[j])
    }
  }
  #sum with Firth correction
  ll = sum(pll) + 0.5*determinant(I, logarithm = TRUE)$modulus[1]
  if (is.nan(ll)){
    ll = -Inf
  }
  
  #Return sum of the log-likelihoods
  return(if (sum == TRUE) ll else (pll + 0.5*determinant(I, logarithm = TRUE)$modulus[1]/length(Y_L15sm)))  
}

# MCMC Setup----
settingsDREAMzs_L15_Hist = list(iterations = 2000000, gamma= NULL, eps = 0, e = 0.05, parallel = NULL, Z = NULL, ZupdateFrequency = 50, pSnooker = 0.1, DEpairs = 2,
                                  nCR = 3, pCRupdate = TRUE, updateInterval = 10,
                                  #burnin must be greater than adaptation.
                                  burnin = 50000, adaptation = 50000, thin = 100, message = FALSE, startValue = 7)

# DREAMzs: prior #2, FixY, PCAcvs, pSensSpec, Historical1s----
dir.create(path = 'DREAMzs_L15_SmithChipVermPrecipPCA_Uncertainty_1915-2020', showWarnings = FALSE)
setUpDREAMzs_L15_pSensSpec_PCAcvs_Historical1s <- createBayesianSetup(likelihood_L15_FixY_pSensSpec_PCAcvs_Historical1s, prior = prior_L15_unif_SensSpec_Hist, parallel = 7, 
                                                                    parallelOptions = list(packages=list('BayesianTools'), 
                                                                                           variables=list('X_L15sm','Y_L15sm', 'predPCAcvs', 'FloodMag_L15sm', 
                                                                                                          'years_L15sm'), dlls=NULL), 
                                                                    names = c('Int', 'PC1', 'PC2', 'pLL', 'pML', 'pSL', 'pUL', 'pNN'), 
                                                                    plotLower = c(-15, -1, -1,0,0,0,0,0), plotUpper = c(0, 6, 6,1,1,1,1,1), 
                                                                    plotBest = c(as.numeric(res_Firth_L15_PC3$coefficients),0.8,0.5,0.2,0.5,0.5))
set.seed(26440)
outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s <- runMCMC(bayesianSetup = setUpDREAMzs_L15_pSensSpec_PCAcvs_Historical1s, sampler = "DREAMzs", settings = settingsDREAMzs_L15_Hist)
#Remove all -Inf in chain. These result from poor initial conditions (prior samples)
outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s = removeInf(outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s)
#Check for remaining infinities
if(any(outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s$InfCk > 0)){
  print('infinities remain within MCMC chain posteriors')
}
summary(outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s)

#Compute WAIC and DIC
outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s$WAICBT = WAIC(outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s, numSamples = 100000)
outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s$WAIC2BT = WAIC2(outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s, numSamples = 100000)
outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s$DIC = DIC(outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s)
#Pointwise log-likelihoods
outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s$pll = outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s$setup$likelihood$density(
  getSample(outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s, parametersOnly = F)[,1:outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s$setup$numPars], sum = FALSE)
#LOO
outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s$LOO = loo(outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s$pll, 
                                                       cores = 7, save_psis = TRUE, r_eff = relative_eff(x = exp(outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s$pll),
                                                                                                         chain_id = c(rep(1,nrow(outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s$chain[[1]])), 
                                                                                                                      rep(2,nrow(outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s$chain[[1]])), 
                                                                                                                      rep(3,nrow(outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s$chain[[1]])), 
                                                                                                                      rep(4,nrow(outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s$chain[[1]])), 
                                                                                                                      rep(5,nrow(outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s$chain[[1]])), 
                                                                                                                      rep(6,nrow(outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s$chain[[1]])), 
                                                                                                                      rep(7,nrow(outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s$chain[[1]])))))
outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s$WAIC = waic(x = outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s$pll)
outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s$WAIC2 = waic2(x = outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s$pll)
#Rejection rate
outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s$rejectRate = rejectionRate(outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s$chain)
#Highest Posterior Density Intervals
outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s$HPD = HPDinterval(outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s$chain)
#Credible Intervals
outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s$CI = getCredibleIntervals(outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s$chain[[1]])
#Effective sample size for each parameter should be similar
outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s$Neff = effectiveSize(outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s$chain[,1:8])

#Sample only the last 1000 from each chain
sample_chain_inds = (nrow(outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s$chain[[1]])-1000):nrow(outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s$chain[[1]])

dir.create('DREAMzs_L15_SmithChipVermPrecipPCA_Uncertainty_1915-2020/MCMC', showWarnings = FALSE)
png('DREAMzs_L15_SmithChipVermPrecipPCA_Uncertainty_1915-2020/MCMC/traceplot2.png', res = 300, units = 'in', width = 7, height = 7)
mcmc_trace(x = outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s$chain[,1:8], size = 0.05)
dev.off()

png('DREAMzs_L15_SmithChipVermPrecipPCA_Uncertainty_1915-2020/MCMC/correlplot.png', res = 300, units = 'in', width = 7, height = 7)
correlationPlot(outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s$chain[sample_chain_inds,1:8], method = 'spearman')
dev.off()

png('DREAMzs_L15_SmithChipVermPrecipPCA_Uncertainty_1915-2020/MCMC/gelmanplot_panel1.png', res = 300, units = 'in', width = 7, height = 7)
gelmanDiagnostics(outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s$chain[,1:6], plot = TRUE, thin = 0)
dev.off()
png('DREAMzs_L15_SmithChipVermPrecipPCA_Uncertainty_1915-2020/MCMC/gelmanplot_panel2.png', res = 300, units = 'in', width = 7, height = 7)
gelmanDiagnostics(outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s$chain[,7:8], plot = TRUE, thin = 0)
dev.off()

png('DREAMzs_L15_SmithChipVermPrecipPCA_Uncertainty_1915-2020/MCMC/marginalplot.png', res = 300, units = 'in', width = 7, height = 7)
marginalPlot(x = outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s$chain[sample_chain_inds,1:8], 
             prior = prior_L15_unif_SensSpec_Hist$sampler(n = 50000), singlePanel = FALSE, 
             xrange = t(rbind(c(-20,5),c(-10,5),c(-5,15), c(0,1), c(0,1), c(0,1), c(0,1), c(0,1))))
dev.off()

png('DREAMzs_L15_SmithChipVermPrecipPCA_Uncertainty_1915-2020/MCMC/PairPlot.png', res = 300, units = 'in', width = 7, height = 7)
mcmc_pairs(outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s$chain[sample_chain_inds,1:8], diag_fun = 'dens', off_diag_fun = 'scatter', 
           off_diag_args = list(size=0.5,alpha=0.5), 
           xlim = rbind(c(-15,0),c(-1,6),c(-1,6),c(0,1),c(0,1),c(0,1),c(0,1),c(0,1),
                        c(-15,0),c(-1,6),c(-1,6),c(0,1),c(0,1),c(0,1),c(0,1),c(0,1),
                        c(-15,0),c(-1,6),c(-1,6),c(0,1),c(0,1),c(0,1),c(0,1),c(0,1),
                        c(-15,0),c(-1,6),c(-1,6),c(0,1),c(0,1),c(0,1),c(0,1),c(0,1),
                        c(-15,0),c(-1,6),c(-1,6),c(0,1),c(0,1),c(0,1),c(0,1),c(0,1),
                        c(-15,0),c(-1,6),c(-1,6),c(0,1),c(0,1),c(0,1),c(0,1),c(0,1),
                        c(-15,0),c(-1,6),c(-1,6),c(0,1),c(0,1),c(0,1),c(0,1),c(0,1),
                        c(-15,0),c(-1,6),c(-1,6),c(0,1),c(0,1),c(0,1),c(0,1),c(0,1)),
           ylim = rbind(c(0,NA),c(-15,0),c(-15,0),c(-15,0),c(-15,0),c(-15,0),c(-15,0),c(-15,0),
                        c(-1,6),c(0,NA),c(-1,6),c(-1,6),c(-1,6),c(-1,6),c(-1,6),c(-1,6),
                        c(-1,6),c(-1,6),c(0,NA),c(-1,6),c(-1,6),c(-1,6),c(-1,6),c(-1,6),
                        c(0,1),c(0,1),c(0,1),c(0,NA),c(0,1),c(0,1),c(0,1),c(0,1),
                        c(0,1),c(0,1),c(0,1),c(0,1),c(0,NA),c(0,1),c(0,1),c(0,1),
                        c(0,1),c(0,1),c(0,1),c(0,1),c(0,1),c(0,NA),c(0,1),c(0,1),
                        c(0,1),c(0,1),c(0,1),c(0,1),c(0,1),c(0,1),c(0,NA),c(0,1),
                        c(0,1),c(0,1),c(0,1),c(0,1),c(0,1),c(0,1),c(0,1),c(0,NA)))
dev.off()

png('DREAMzs_L15_SmithChipVermPrecipPCA_Uncertainty_1915-2020/MCMC/autocorrplot_panel1.png', res = 300, units = 'in', width = 7, height = 7)
autocorr.plot(x = outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s$chain[[1]][-1,1:8], lag.max = 100)
dev.off()

png('DREAMzs_L15_SmithChipVermPrecipPCA_Uncertainty_1915-2020/MCMC/autocorrplot2.png', res = 300, units = 'in', width = 7, height = 7)
mcmc_acf(x = outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s$chain, 
         pars = outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s$setup$names, lags = 20)
dev.off()

png('DREAMzs_L15_SmithChipVermPrecipPCA_Uncertainty_1915-2020/MCMC/densOverlay.png', res = 300, units = 'in', width = 7, height = 7)
mcmc_dens_overlay(outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s$chain[sample_chain_inds,])
dev.off()

png('DREAMzs_L15_SmithChipVermPrecipPCA_Uncertainty_1915-2020/MCMC/HPD.png', res = 300, units = 'in', width = 7, height = 7)
mcmc_areas(x = outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s$chain[sample_chain_inds,], 
           pars = outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s$setup$names, 
           point_est = 'mean', prob = 0.95)
dev.off()

png('DREAMzs_L15_SmithChipVermPrecipPCA_Uncertainty_1915-2020/MCMC/HPD_boxplot.png', res = 300, units = 'in', width = 7, height = 7)
mcmc_intervals(x = outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s$chain[sample_chain_inds,], 
               pars = outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s$setup$names,
               prob_outer = 0.95)
dev.off()

geweke.plot(x = outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s$chain[,1:8], ask = FALSE, nbins = 8)

stopParallel(setUpDREAMzs_L15_pSensSpec_PCAcvs_Historical1s)

# Figure 3----
pdf('DREAMzs_L15_SmithChipVermPrecipPCA_Uncertainty_1915-2020/MCMC/Fig3-densOverlay.pdf', width = 7, height = 7)
mcmc_dens_overlay(outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s$chain[sample_chain_inds,1:8], color_chains = FALSE)
dev.off()

# DREAMzs: prior #2, FixY, PCAcvs, pSensSpec, Historical1s, pAll----
dir.create(path = 'DREAMzs_L15_SmithChipVermPrecipPCA_Uncertainty_1915-2020_pAll', showWarnings = FALSE)
setUpDREAMzs_L15_pSensSpec_PCAcvs_Historical1s_pAll <- createBayesianSetup(likelihood_L15_FixY_pSensSpec_PCAcvs_Historical1s_pAll, prior = prior_L15_unif_SensSpec_pAll, parallel = 7, 
                                                                        parallelOptions = list(packages=list('BayesianTools'), 
                                                                                               variables=list('X_L15sm','Y_L15sm', 'predPCAcvs', 'FloodMag_L15sm', 
                                                                                                              'years_L15sm'), dlls=NULL), 
                                                                        names = c('Int', 'PC1', 'PC2', paste0('pLL',seq(1,6,1)), paste0('pML',seq(1,5,1)), 
                                                                                  paste0('pSL',seq(1,5,1)), paste0('pUL',seq(1,4,1)), 'pNN'), 
                                                                        plotLower = c(-15, -6, 0,rep(0,21)), plotUpper = c(0, 2, 6,rep(1,21)), 
                                                                        plotBest = c(as.numeric(res_Firth$coefficients),rep(0.5,21)))
set.seed(26441)
outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s_pAll <- runMCMC(bayesianSetup = setUpDREAMzs_L15_pSensSpec_PCAcvs_Historical1s_pAll, sampler = "DREAMzs", settings = settingsDREAMzs_L15_Hist)
#Remove all -Inf in chain. These result from poor initial conditions (prior samples)
outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s_pAll = removeInf(outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s_pAll)
#Check for remaining infinities
if(any(outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s_pAll$InfCk > 0)){
  print('infinities remain within MCMC chain posteriors')
}
summary(outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s_pAll)

#Compute WAIC and DIC
outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s_pAll$WAICBT = WAIC(outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s_pAll, numSamples = 100000)
outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s_pAll$WAIC2BT = WAIC2(outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s_pAll, numSamples = 100000)
outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s_pAll$DIC = DIC(outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s_pAll)
#Pointwise log-likelihoods
outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s_pAll$pll = outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s_pAll$setup$likelihood$density(
  getSample(outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s_pAll, parametersOnly = F)[,1:outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s_pAll$setup$numPars], sum = FALSE)
#LOO
outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s_pAll$LOO = loo(outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s_pAll$pll, 
                                                            cores = 7, save_psis = TRUE, r_eff = relative_eff(x = exp(outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s_pAll$pll), 
                                                                                                              chain_id = c(rep(1,nrow(outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s_pAll$chain[[1]])), 
                                                                                                                           rep(2,nrow(outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s_pAll$chain[[1]])), 
                                                                                                                           rep(3,nrow(outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s_pAll$chain[[1]])), 
                                                                                                                           rep(4,nrow(outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s_pAll$chain[[1]])), 
                                                                                                                           rep(5,nrow(outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s_pAll$chain[[1]])), 
                                                                                                                           rep(6,nrow(outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s_pAll$chain[[1]])), 
                                                                                                                           rep(7,nrow(outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s_pAll$chain[[1]])))))
outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s_pAll$WAIC = waic(x = outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s_pAll$pll)
outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s_pAll$WAIC2 = waic2(x = outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s_pAll$pll)
#Rejection rate
outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s_pAll$rejectRate = rejectionRate(outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s_pAll$chain)
#Highest Posterior Density Intervals
outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s_pAll$HPD = HPDinterval(outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s_pAll$chain)
#Credible Intervals
outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s_pAll$CI = getCredibleIntervals(outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s_pAll$chain[[1]])
#Effective sample size for each parameter should be similar
outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s_pAll$Neff = effectiveSize(outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s_pAll$chain[,1:24])

#Sample only the last 1000 from each chain
sample_chain_inds = (nrow(outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s_pAll$chain[[1]])-1000):nrow(outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s_pAll$chain[[1]])

dir.create('DREAMzs_L15_SmithChipVermPrecipPCA_Uncertainty_1915-2020_pAll/MCMC', showWarnings = FALSE)
png('DREAMzs_L15_SmithChipVermPrecipPCA_Uncertainty_1915-2020_pAll/MCMC/traceplot2_panel1.png', res = 300, units = 'in', width = 7, height = 7)
mcmc_trace(x = outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s_pAll$chain[,1:9], size = 0.05)
dev.off()
png('DREAMzs_L15_SmithChipVermPrecipPCA_Uncertainty_1915-2020_pAll/MCMC/traceplot2_panel2.png', res = 300, units = 'in', width = 7, height = 7)
mcmc_trace(x = outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s_pAll$chain[,10:14], size = 0.05)
dev.off()
png('DREAMzs_L15_SmithChipVermPrecipPCA_Uncertainty_1915-2020_pAll/MCMC/traceplot2_panel3.png', res = 300, units = 'in', width = 7, height = 7)
mcmc_trace(x = outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s_pAll$chain[,15:19], size = 0.05)
dev.off()
png('DREAMzs_L15_SmithChipVermPrecipPCA_Uncertainty_1915-2020_pAll/MCMC/traceplot2_panel4.png', res = 300, units = 'in', width = 7, height = 7)
mcmc_trace(x = outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s_pAll$chain[,20:24], size = 0.05)
dev.off()

png('DREAMzs_L15_SmithChipVermPrecipPCA_Uncertainty_1915-2020_pAll/MCMC/correlplot.png', res = 300, units = 'in', width = 7, height = 7)
correlationPlot(outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s_pAll$chain[sample_chain_inds,1:24], method = 'spearman')
dev.off()

png('DREAMzs_L15_SmithChipVermPrecipPCA_Uncertainty_1915-2020_pAll/MCMC/gelmanplot_panel1.png', res = 300, units = 'in', width = 7, height = 7)
gelmanDiagnostics(outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s_pAll$chain[,1:6], plot = TRUE, thin = 0)
dev.off()
png('DREAMzs_L15_SmithChipVermPrecipPCA_Uncertainty_1915-2020_pAll/MCMC/gelmanplot_panel2.png', res = 300, units = 'in', width = 7, height = 7)
gelmanDiagnostics(outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s_pAll$chain[,7:12], plot = TRUE, thin = 0)
dev.off()
png('DREAMzs_L15_SmithChipVermPrecipPCA_Uncertainty_1915-2020_pAll/MCMC/gelmanplot_panel3.png', res = 300, units = 'in', width = 7, height = 7)
gelmanDiagnostics(outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s_pAll$chain[,13:18], plot = TRUE, thin = 0)
dev.off()
png('DREAMzs_L15_SmithChipVermPrecipPCA_Uncertainty_1915-2020_pAll/MCMC/gelmanplot_panel4.png', res = 300, units = 'in', width = 7, height = 7)
gelmanDiagnostics(outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s_pAll$chain[,19:24], plot = TRUE, thin = 0)
dev.off()

png('DREAMzs_L15_SmithChipVermPrecipPCA_Uncertainty_1915-2020_pAll/MCMC/marginalplot_panel1.png', res = 300, units = 'in', width = 7, height = 7)
marginalPlot(x = outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s_pAll$chain[sample_chain_inds,1:9], prior = prior_L15_unif_SensSpec_pAll$sampler(n = 50000)[,1:9], 
             singlePanel = FALSE, xrange = t(rbind(c(-20,5),c(-10,5),c(-5,15), c(0,1), c(0,1), c(0,1), c(0,1), c(0,1), c(0,1))))
dev.off()
png('DREAMzs_L15_SmithChipVermPrecipPCA_Uncertainty_1915-2020_pAll/MCMC/marginalplot_panel2.png', res = 300, units = 'in', width = 7, height = 7)
marginalPlot(x = outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s_pAll$chain[sample_chain_inds,10:14], prior = prior_L15_unif_SensSpec_pAll$sampler(n = 50000)[,10:14], 
             singlePanel = FALSE, xrange = t(rbind(c(0,1), c(0,1), c(0,1), c(0,1), c(0,1))))
dev.off()
png('DREAMzs_L15_SmithChipVermPrecipPCA_Uncertainty_1915-2020_pAll/MCMC/marginalplot_panel3.png', res = 300, units = 'in', width = 7, height = 7)
marginalPlot(x = outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s_pAll$chain[sample_chain_inds,15:19], prior = prior_L15_unif_SensSpec_pAll$sampler(n = 50000)[,15:19], 
             singlePanel = FALSE, xrange = t(rbind(c(0,1), c(0,1), c(0,1), c(0,1), c(0,1))))
dev.off()
png('DREAMzs_L15_SmithChipVermPrecipPCA_Uncertainty_1915-2020_pAll/MCMC/marginalplot_panel4.png', res = 300, units = 'in', width = 7, height = 7)
marginalPlot(x = outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s_pAll$chain[sample_chain_inds,20:24], prior = prior_L15_unif_SensSpec_pAll$sampler(n = 50000)[,20:24], 
             singlePanel = FALSE, xrange = t(rbind(c(0,1), c(0,1), c(0,1), c(0,1), c(0,1))))
dev.off()


png('DREAMzs_L15_SmithChipVermPrecipPCA_Uncertainty_1915-2020_pAll/MCMC/PairPlot.png', res = 300, units = 'in', width = 7, height = 7)
mcmc_pairs(outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s_pAll$chain[sample_chain_inds,1:8], diag_fun = 'dens', off_diag_fun = 'scatter', 
           off_diag_args = list(size=0.5,alpha=0.5), 
           xlim = rbind(c(-15,0),c(-6,1),c(0,6),c(0,1),c(0,1),c(0,1),c(0,1),c(0,1),
                        c(-15,0),c(-6,1),c(0,6),c(0,1),c(0,1),c(0,1),c(0,1),c(0,1),
                        c(-15,0),c(-6,1),c(0,6),c(0,1),c(0,1),c(0,1),c(0,1),c(0,1),
                        c(-15,0),c(-6,1),c(0,6),c(0,1),c(0,1),c(0,1),c(0,1),c(0,1),
                        c(-15,0),c(-6,1),c(0,6),c(0,1),c(0,1),c(0,1),c(0,1),c(0,1),
                        c(-15,0),c(-6,1),c(0,6),c(0,1),c(0,1),c(0,1),c(0,1),c(0,1),
                        c(-15,0),c(-6,1),c(0,6),c(0,1),c(0,1),c(0,1),c(0,1),c(0,1),
                        c(-15,0),c(-6,1),c(0,6),c(0,1),c(0,1),c(0,1),c(0,1),c(0,1)),
           ylim = rbind(c(0,NA),c(-15,0),c(-15,0),c(-15,0),c(-15,0),c(-15,0),c(-15,0),c(-15,0),
                        c(-6,1),c(0,NA),c(-6,1),c(-6,1),c(-6,1),c(-6,1),c(-6,1),c(-6,1),
                        c(0,6),c(0,6),c(0,NA),c(0,6),c(0,6),c(0,6),c(0,6),c(0,6),
                        c(0,1),c(0,1),c(0,1),c(0,NA),c(0,1),c(0,1),c(0,1),c(0,1),
                        c(0,1),c(0,1),c(0,1),c(0,1),c(0,NA),c(0,1),c(0,1),c(0,1),
                        c(0,1),c(0,1),c(0,1),c(0,1),c(0,1),c(0,NA),c(0,1),c(0,1),
                        c(0,1),c(0,1),c(0,1),c(0,1),c(0,1),c(0,1),c(0,NA),c(0,1),
                        c(0,1),c(0,1),c(0,1),c(0,1),c(0,1),c(0,1),c(0,1),c(0,NA)))
dev.off()

png('DREAMzs_L15_SmithChipVermPrecipPCA_Uncertainty_1915-2020_pAll/MCMC/autocorrplot_panel1.png', res = 300, units = 'in', width = 7, height = 7)
autocorr.plot(x = outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s_pAll$chain[[1]][-1,1:9], lag.max = 100)
dev.off()
png('DREAMzs_L15_SmithChipVermPrecipPCA_Uncertainty_1915-2020_pAll/MCMC/autocorrplot_panel2.png', res = 300, units = 'in', width = 7, height = 7)
autocorr.plot(x = outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s_pAll$chain[[1]][-1,10:14], lag.max = 100)
dev.off()
png('DREAMzs_L15_SmithChipVermPrecipPCA_Uncertainty_1915-2020_pAll/MCMC/autocorrplot_panel3.png', res = 300, units = 'in', width = 7, height = 7)
autocorr.plot(x = outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s_pAll$chain[[1]][-1,15:19], lag.max = 100)
dev.off()
png('DREAMzs_L15_SmithChipVermPrecipPCA_Uncertainty_1915-2020_pAll/MCMC/autocorrplot_panel4.png', res = 300, units = 'in', width = 7, height = 7)
autocorr.plot(x = outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s_pAll$chain[[1]][-1,20:24], lag.max = 100)
dev.off()

png('DREAMzs_L15_SmithChipVermPrecipPCA_Uncertainty_1915-2020_pAll/MCMC/autocorrplot2.png', res = 300, units = 'in', width = 7, height = 7)
mcmc_acf(x = outDREAMzs__L15_pSensSpec_PCAcvs_Historical1s_pAll$chain, 
         pars = outDREAMzs__L15_pSensSpec_PCAcvs_Historical1s_pAll$setup$names, lags = 20)
dev.off()

png('DREAMzs_L15_SmithChipVermPrecipPCA_Uncertainty_1915-2020_pAll/MCMC/densOverlay.png', res = 300, units = 'in', width = 7, height = 7)
mcmc_dens_overlay(outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s_pAll$chain[sample_chain_inds,])
dev.off()

png('DREAMzs_L15_SmithChipVermPrecipPCA_Uncertainty_1915-2020_pAll/MCMC/HPD.png', res = 300, units = 'in', width = 7, height = 7)
mcmc_areas(x = outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s_pAll$chain[sample_chain_inds,], 
           pars = outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s_pAll$setup$names, 
           point_est = 'mean', prob = 0.95)
dev.off()

png('DREAMzs_L15_SmithChipVermPrecipPCA_Uncertainty_1915-2020_pAll/MCMC/HPD_boxplot.png', res = 300, units = 'in', width = 7, height = 7)
mcmc_intervals(x = outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s_pAll$chain[sample_chain_inds,], 
               pars = outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s_pAll$setup$names,
               prob_outer = 0.95)
dev.off()

geweke.plot(x = outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s_pAll$chain[,1:24], ask = FALSE, nbins = 8)

stopParallel(setUpDREAMzs_L15_pSensSpec_PCAcvs_Historical1s_pAll)

# DREAMzs: prior #2, FixY, PCAcvs, pSensSpec, Historical1s, pAll, theta----
dir.create(path = 'DREAMzs_L15_SmithChipVermPrecipPCA_Uncertainty_1915-2020_pAll_theta', showWarnings = FALSE)
setUpDREAMzs_L15_pSensSpec_PCAcvs_Historical1s_pAll_theta <- 
  createBayesianSetup(likelihood_L15_FixY_pSensSpec_PCAcvs_Historical1s_pAll_theta, 
                      prior = prior_L15_unif_SensSpec_pAll_theta, 
                      parallel = 7, parallelOptions = list(packages=list('BayesianTools'),
                                                           variables=list('X_L15sm','Y_L15sm', 'predPCAcvs', 
                                                                          'FloodMag_L15sm', 'years_L15sm'), dlls=NULL), 
                                                                        
                      names = c('Int', 'PC1', 'PC2', paste0('pLL',seq(1,6,1)), paste0('pML',seq(1,5,1)), 
                                paste0('pSL',seq(1,5,1)), paste0('pUL',seq(1,4,1)), paste0('pNN',seq(1,23,1))), 
                      plotLower = c(-15, -6, 0,rep(0,43)), plotUpper = c(0, 2, 6,rep(1,43)), 
                      plotBest = c(as.numeric(res_Firth$coefficients),rep(0.5,43)))
set.seed(31422)
outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s_pAll_theta = 
  runMCMC(bayesianSetup = setUpDREAMzs_L15_pSensSpec_PCAcvs_Historical1s_pAll_theta,
          sampler = "DREAMzs", settings = settingsDREAMzs_L15_Hist)
#Remove all -Inf in chain. These result from poor initial conditions (prior samples)
outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s_pAll_theta = 
  removeInf(outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s_pAll_theta)
#Check for remaining infinities
if(any(outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s_pAll_theta$InfCk > 0)){
  print('infinities remain within MCMC chain posteriors')
}
summary(outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s_pAll_theta)

#Compute WAIC and DIC
outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s_pAll_theta$WAICBT = 
  WAIC(outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s_pAll_theta, numSamples = 100000)
outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s_pAll_theta$WAIC2BT = 
  WAIC2(outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s_pAll_theta, numSamples = 100000)
outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s_pAll_theta$DIC = 
  DIC(outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s_pAll_theta)
#Pointwise log-likelihoods
outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s_pAll_theta$pll = 
  outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s_pAll_theta$setup$likelihood$density(
    getSample(outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s_pAll_theta,                       
              parametersOnly = F)[,1:outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s_pAll_theta$setup$numPars],
    sum = FALSE)
#LOO
outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s_pAll_theta$LOO = 
  loo(outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s_pAll_theta$pll,
      cores = 7, save_psis = TRUE, 
      r_eff = relative_eff(x = exp(outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s_pAll_theta$pll), 
                           chain_id = c(rep(1,nrow(outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s_pAll_theta$chain[[1]])),         
                                        rep(2,nrow(outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s_pAll_theta$chain[[1]])),         
                                        rep(3,nrow(outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s_pAll_theta$chain[[1]])),         
                                        rep(4,nrow(outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s_pAll_theta$chain[[1]])),         
                                        rep(5,nrow(outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s_pAll_theta$chain[[1]])),         
                                        rep(6,nrow(outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s_pAll_theta$chain[[1]])),         
                                        rep(7,nrow(outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s_pAll_theta$chain[[1]])))))
outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s_pAll_theta$WAIC = 
  waic(x = outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s_pAll_theta$pll)
outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s_pAll_theta$WAIC2 = 
  waic2(x = outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s_pAll_theta$pll)
#Rejection rate
outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s_pAll_theta$rejectRate = 
  rejectionRate(outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s_pAll_theta$chain)
#Highest Posterior Density Intervals
outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s_pAll_theta$HPD = 
  HPDinterval(outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s_pAll_theta$chain)
#Credible Intervals
outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s_pAll_theta$CI = 
  getCredibleIntervals(outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s_pAll_theta$chain[[1]])
#Effective sample size for each parameter should be similar
outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s_pAll_theta$Neff = 
  effectiveSize(outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s_pAll_theta$chain[,1:46])

#Sample only the last 1000 from each chain
sample_chain_inds = (nrow(outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s_pAll_theta$chain[[1]])-1000):nrow(outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s_pAll_theta$chain[[1]])

dir.create('DREAMzs_L15_SmithChipVermPrecipPCA_Uncertainty_1915-2020_pAll_theta/MCMC', showWarnings = FALSE)
png('DREAMzs_L15_SmithChipVermPrecipPCA_Uncertainty_1915-2020_pAll_theta/MCMC/traceplot2.png', res = 300, units = 'in', 
    width = 7, height = 7)
mcmc_trace(x = outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s_pAll_theta$chain[,1:46], size = 0.05)
dev.off()

png('DREAMzs_L15_SmithChipVermPrecipPCA_Uncertainty_1915-2020_pAll_theta/MCMC/correlplot.png', res = 300, units = 'in',
    width = 14, height = 14)
correlationPlot(outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s_pAll_theta$chain[sample_chain_inds,1:46], method = 'spearman')
dev.off()

png('DREAMzs_L15_SmithChipVermPrecipPCA_Uncertainty_1915-2020_pAll_theta/MCMC/gelmanplot_panel1.png', res = 300, units = 'in', 
    width = 7, height = 7)
gelmanDiagnostics(outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s_pAll_theta$chain[,1:6], plot = TRUE, thin = 0)
dev.off()
png('DREAMzs_L15_SmithChipVermPrecipPCA_Uncertainty_1915-2020_pAll_theta/MCMC/gelmanplot_panel2.png', res = 300, units = 'in', 
    width = 7, height = 7)
gelmanDiagnostics(outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s_pAll_theta$chain[,7:12], plot = TRUE, thin = 0)
dev.off()
png('DREAMzs_L15_SmithChipVermPrecipPCA_Uncertainty_1915-2020_pAll_theta/MCMC/gelmanplot_panel3.png', res = 300, units = 'in', 
    width = 7, height = 7)
gelmanDiagnostics(outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s_pAll_theta$chain[,13:18], plot = TRUE, thin = 0)
dev.off()
png('DREAMzs_L15_SmithChipVermPrecipPCA_Uncertainty_1915-2020_pAll_theta/MCMC/gelmanplot_panel4.png', res = 300, units = 'in', 
    width = 7, height = 7)
gelmanDiagnostics(outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s_pAll_theta$chain[,19:24], plot = TRUE, thin = 0)
dev.off()
png('DREAMzs_L15_SmithChipVermPrecipPCA_Uncertainty_1915-2020_pAll_theta/MCMC/gelmanplot_panel5.png', res = 300, units = 'in', 
    width = 7, height = 7)
gelmanDiagnostics(outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s_pAll_theta$chain[,25:30], plot = TRUE, thin = 0)
dev.off()
png('DREAMzs_L15_SmithChipVermPrecipPCA_Uncertainty_1915-2020_pAll_theta/MCMC/gelmanplot_panel6.png', res = 300, units = 'in', 
    width = 7, height = 7)
gelmanDiagnostics(outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s_pAll_theta$chain[,31:36], plot = TRUE, thin = 0)
dev.off()
png('DREAMzs_L15_SmithChipVermPrecipPCA_Uncertainty_1915-2020_pAll_theta/MCMC/gelmanplot_panel7.png', res = 300, units = 'in', 
    width = 7, height = 7)
gelmanDiagnostics(outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s_pAll_theta$chain[,37:42], plot = TRUE, thin = 0)
dev.off()
png('DREAMzs_L15_SmithChipVermPrecipPCA_Uncertainty_1915-2020_pAll_theta/MCMC/gelmanplot_panel8.png', res = 300, units = 'in', 
    width = 7, height = 7)
gelmanDiagnostics(outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s_pAll_theta$chain[,43:46], plot = TRUE, thin = 0)
dev.off()

png('DREAMzs_L15_SmithChipVermPrecipPCA_Uncertainty_1915-2020_pAll_theta/MCMC/marginalplot_panel1.png', res = 300, units = 'in', 
    width = 7, height = 7)
marginalPlot(x = outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s_pAll_theta$chain[sample_chain_inds,1:9], 
             prior = prior_L15_unif_SensSpec_pAll_theta$sampler(n = 50000)[,1:9], singlePanel = FALSE, 
             xrange = t(rbind(c(-20,5),c(-10,5),c(-5,15), c(0,1), c(0,1), c(0,1), c(0,1), c(0,1), c(0,1))))
dev.off()
png('DREAMzs_L15_SmithChipVermPrecipPCA_Uncertainty_1915-2020_pAll_theta/MCMC/marginalplot_panel2.png', res = 300, units = 'in', 
    width = 7, height = 7)
marginalPlot(x = outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s_pAll_theta$chain[sample_chain_inds,10:18], 
             prior = prior_L15_unif_SensSpec_pAll_theta$sampler(n = 50000)[,10:18], singlePanel = FALSE, 
             xrange = t(rbind(c(0,1), c(0,1), c(0,1), c(0,1), c(0,1), c(0,1), c(0,1), c(0,1), c(0,1))))
dev.off()
png('DREAMzs_L15_SmithChipVermPrecipPCA_Uncertainty_1915-2020_pAll_theta/MCMC/marginalplot_panel3.png', res = 300, units = 'in', 
    width = 7, height = 7)
marginalPlot(x = outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s_pAll_theta$chain[sample_chain_inds,19:27], 
             prior = prior_L15_unif_SensSpec_pAll_theta$sampler(n = 50000)[,19:27], singlePanel = FALSE, 
             xrange = t(rbind(c(0,1), c(0,1), c(0,1), c(0,1), c(0,1), c(0,1), c(0,1), c(0,1), c(0,1))))
dev.off()
png('DREAMzs_L15_SmithChipVermPrecipPCA_Uncertainty_1915-2020_pAll_theta/MCMC/marginalplot_panel4.png', res = 300, units = 'in', 
    width = 7, height = 7)
marginalPlot(x = outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s_pAll_theta$chain[sample_chain_inds,28:36], 
             prior = prior_L15_unif_SensSpec_pAll_theta$sampler(n = 50000)[,28:36], singlePanel = FALSE, 
             xrange = t(rbind(c(0,1), c(0,1), c(0,1), c(0,1), c(0,1), c(0,1), c(0,1), c(0,1), c(0,1))))
dev.off()
png('DREAMzs_L15_SmithChipVermPrecipPCA_Uncertainty_1915-2020_pAll_theta/MCMC/marginalplot_panel5.png', res = 300, units = 'in', 
    width = 7, height = 7)
marginalPlot(x = outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s_pAll_theta$chain[sample_chain_inds,37:45], 
             prior = prior_L15_unif_SensSpec_pAll_theta$sampler(n = 50000)[,37:45], singlePanel = FALSE, 
             xrange = t(rbind(c(0,1), c(0,1), c(0,1), c(0,1), c(0,1), c(0,1), c(0,1), c(0,1), c(0,1))))
dev.off()
png('DREAMzs_L15_SmithChipVermPrecipPCA_Uncertainty_1915-2020_pAll_theta/MCMC/marginalplot_panel6.png', res = 300, units = 'in', 
    width = 7, height = 7)
marginalPlot(x = outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s_pAll_theta$chain[sample_chain_inds,45:46], 
             prior = prior_L15_unif_SensSpec_pAll_theta$sampler(n = 50000)[,45:46], singlePanel = FALSE, 
             xrange = t(rbind(c(0,1), c(0,1))))
dev.off()

png('DREAMzs_L15_SmithChipVermPrecipPCA_Uncertainty_1915-2020_pAll_theta/MCMC/PairPlot.png', res = 300, units = 'in', 
    width = 7, height = 7)
mcmc_pairs(outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s_pAll_theta$chain[sample_chain_inds,1:8], diag_fun = 'dens', 
           off_diag_fun = 'scatter', 
           off_diag_args = list(size=0.5,alpha=0.5), 
           xlim = rbind(c(-15,0),c(-1,6),c(-1,6),c(0,1),c(0,1),c(0,1),c(0,1),c(0,1),
                        c(-15,0),c(-1,6),c(-1,6),c(0,1),c(0,1),c(0,1),c(0,1),c(0,1),
                        c(-15,0),c(-1,6),c(-1,6),c(0,1),c(0,1),c(0,1),c(0,1),c(0,1),
                        c(-15,0),c(-1,6),c(-1,6),c(0,1),c(0,1),c(0,1),c(0,1),c(0,1),
                        c(-15,0),c(-1,6),c(-1,6),c(0,1),c(0,1),c(0,1),c(0,1),c(0,1),
                        c(-15,0),c(-1,6),c(-1,6),c(0,1),c(0,1),c(0,1),c(0,1),c(0,1),
                        c(-15,0),c(-1,6),c(-1,6),c(0,1),c(0,1),c(0,1),c(0,1),c(0,1),
                        c(-15,0),c(-1,6),c(-1,6),c(0,1),c(0,1),c(0,1),c(0,1),c(0,1)),
           ylim = rbind(c(0,NA),c(-15,0),c(-15,0),c(-15,0),c(-15,0),c(-15,0),c(-15,0),c(-15,0),
                        c(-1,6),c(0,NA),c(-1,6),c(-1,6),c(-1,6),c(-1,6),c(-1,6),c(-1,6),
                        c(-1,6),c(-1,6),c(0,NA),c(-1,6),c(-1,6),c(-1,6),c(-1,6),c(-1,6),
                        c(0,1),c(0,1),c(0,1),c(0,NA),c(0,1),c(0,1),c(0,1),c(0,1),
                        c(0,1),c(0,1),c(0,1),c(0,1),c(0,NA),c(0,1),c(0,1),c(0,1),
                        c(0,1),c(0,1),c(0,1),c(0,1),c(0,1),c(0,NA),c(0,1),c(0,1),
                        c(0,1),c(0,1),c(0,1),c(0,1),c(0,1),c(0,1),c(0,NA),c(0,1),
                        c(0,1),c(0,1),c(0,1),c(0,1),c(0,1),c(0,1),c(0,1),c(0,NA)))
dev.off()

png('DREAMzs_L15_SmithChipVermPrecipPCA_Uncertainty_1915-2020_pAll_theta/MCMC/autocorrplot2.png', res = 300, units = 'in', width = 7, height = 7)
mcmc_acf(x = outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s_pAll_theta$chain, 
         pars = outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s_pAll_theta$setup$names, lags = 100)
dev.off()

png('DREAMzs_L15_SmithChipVermPrecipPCA_Uncertainty_1915-2020_pAll_theta/MCMC/densOverlay.png', res = 300, units = 'in', 
    width = 7, height = 7)
mcmc_dens_overlay(outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s_pAll_theta$chain[sample_chain_inds,])
dev.off()

png('DREAMzs_L15_SmithChipVermPrecipPCA_Uncertainty_1915-2020_pAll_theta/MCMC/HPD.png', res = 300, units = 'in', width = 7, height = 7)
mcmc_areas(x = outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s_pAll_theta$chain[sample_chain_inds,], 
           pars = outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s_pAll_theta$setup$names, 
           point_est = 'mean', prob = 0.95)
dev.off()

png('DREAMzs_L15_SmithChipVermPrecipPCA_Uncertainty_1915-2020_pAll_theta/MCMC/HPD_boxplot.png', res = 300, units = 'in', width = 7, height = 7)
mcmc_intervals(x = outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s_pAll_theta$chain[sample_chain_inds,], 
               pars = outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s_pAll_theta$setup$names,
               prob_outer = 0.95)
dev.off()

geweke.plot(x = outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s_pAll_theta$chain[,1:46], ask = FALSE, nbins = 8)

stopParallel(setUpDREAMzs_L15_pSensSpec_PCAcvs_Historical1s_pAll_theta)

#Sample from posterior, model with uncertainty----
ppSample = function(BTout, #output from a BayesianTools MCMC 
                    n = 1000,     #number of posterior samples to draw from each chain
                    Xp,      #the X data to use in regression 
                    FloodMag,#the flood magnitude data
                    years,    #the year data
                    Yp,        #the observed flood data
                    seed     #the random seed to use
                    ){
  #Ensure n is less than length of chains
  if (n > nrow(BTout$chain[[1]])){
    stop('n must be less than or equal to the number of samples in each chain')
  }
  
  set.seed(seed)
  
  #Get the last n samples from each chain
  param = matrix(NA, nrow = n*length(BTout$chain), ncol = BTout$setup$numPars)
  for (i in 1:length(BTout$chain)){
    param[(1+(n*(i-1))):(n*i),] = BTout$chain[[i]][(nrow(BTout$chain[[i]])-(n-1)):nrow(BTout$chain[[i]]), 
                                                   1:BTout$setup$numPars]
  }
  
  #Get predicted y (log odds) for each of the parameter sets
  #numerator
  num = t(apply(X = param, MARGIN = 1, FUN = function(p){exp(p[1] + p[2]*Xp[,1] + p[3]*Xp[,2])} ))
  #predicted probability of large IJF
  ppi = t(apply(X = num, MARGIN = 1, FUN = function(num){num/(1+num)}))
  
  #Determine indices of historical large Y = 1
  IndL = which((FloodMag == 'L') & (years < 1962))
  #Determine indices of historical moderate Y = 1
  IndM = which((FloodMag == 'M') & (years < 1962))
  #Determine indices of historical small Y = 1
  IndS = which((FloodMag == 'S') & (years < 1962))
  #Determine indices of historical not a flood Y = 0
  IndN = which((FloodMag == 'N') & (years < 1962))
  #Same for floods of unknown magnitude Y = 1
  IndU = which((FloodMag == 'U'))
  
  #Edit the historical predictions based on the probability of misclassification
  qi = ppi
  for(i in 1:nrow(qi)){
    qi[i,IndL] = ppi[i,IndL]*param[i,4] + (1-ppi[i,IndL])*(1-param[i,8])
    qi[i,IndM] = ppi[i,IndM]*param[i,5] + (1-ppi[i,IndM])*(1-param[i,8])
    qi[i,IndS] = ppi[i,IndS]*param[i,6] + (1-ppi[i,IndS])*(1-param[i,8])
    qi[i,IndU] = ppi[i,IndU]*param[i,7] + (1-ppi[i,IndU])*(1-param[i,8])
    #Weighted eta
    s11 = (param[i,5]*length(IndM) + param[i,6]*length(IndS) + param[i,7]*length(IndU) + param[i,4]*length(IndL))/(
      length(IndM) + length(IndS) + length(IndU) + length(IndL))
    qi[i,IndN] = ppi[i,IndN]*s11 + (1-ppi[i,IndN])*(1-param[i,8])
  }
  
  #Sample a matrix of possible y values based on qi probability of IJF in historical years, and append the 1962-present years
  y = matrix(NA, nrow = nrow(qi), ncol = ncol(qi))
  for (i in 1:ncol(y[,years<1963])){
    y[,i] = rbinom(n = nrow(y), size = 1, prob = qi[,i])
  }
  for (i in 1:ncol(y[,years>=1963])){
    y[,which(years>=1963)[i]] = Yp[years>=1963][i]
  }
    
  #Sample a vector of yrep based on the regression coefficients
  yrep = matrix(NA, nrow = nrow(ppi), ncol = ncol(ppi))
  for (i in 1:ncol(yrep)){
    yrep[,i] = rbinom(n = nrow(yrep), size = 1, prob = ppi[,i])
  }
    
  #return list of params, y and yrep
  retl = list(param=param, pi=ppi, qi=qi, y=y, yrep=yrep)
  return(retl)
}

#Posterior Predictive Check for best model
ppc = ppSample(BTout = outDREAMzs_L15_pSensSpec_PCAcvs_Historical1s, n = 143, Xp = predPCAcvs, 
               FloodMag = FloodMag_L15sm, years = years_L15sm, Yp = Y_L15sm, seed = 34)
bayesplot::ppc_bars(y = ppc$y[1,44:93], yrep = ppc$yrep[,44:93])
#Need to edit this function to plot uncertain y as well as uncertainty in yrep
#Vector of sum for y
ySum = apply(X = ppc$y, MARGIN = 1, FUN = sum)
#Quantiles of sum for y
ySum2p5 = quantile(x = ySum, probs = 0.025)
ySum97p5 = quantile(x = ySum, probs = 0.975)
ySum2p5_0 = quantile(x = ncol(ppc$y)-ySum, probs = 0.025)
ySum97p5_0 = quantile(x = ncol(ppc$y)-ySum, probs = 0.975)
#Vector of sum for yrep
yrepSum = apply(X = ppc$yrep, MARGIN = 1, FUN = sum)
#Quantiles of sum for y
yrepSum2p5 = quantile(x = yrepSum, probs = 0.025)
yrepSum97p5 = quantile(x = yrepSum, probs = 0.975)
yrepSum2p5_0 = quantile(x = ncol(ppc$y)-yrepSum, probs = 0.025)
yrepSum97p5_0 = quantile(x = ncol(ppc$y)-yrepSum, probs = 0.975)

png('DREAMzs_L15_SmithChipVermPrecipPCA_Uncertainty_1915-2020/MCMC/ppcbars.png', res=300, height = 5, width = 5, units = 'in')
#plot bars for the uncertain y
barplot(height = c(mean(ncol(ppc$y)-ySum), mean(ySum)), space = 0, names.arg = c(0,1), col = 'gray', 
        border = 'white', ylim = c(0,100))
par(new=TRUE)
#plot bars for the certain y in a darker shade
barplot(height = c(length(which(ppc$y[1,44:93] == 0)), length(which(ppc$y[1,44:93] == 1))), space = 0, 
        col = 'black', border = 'white', ylim = c(0,100))
#Add error bars for y and yrep on each
arrows(x0 = c(0.25,1.25), y0 = c(ySum2p5_0, ySum2p5), x1 = c(0.25,1.25), y1 = c(ySum97p5_0, ySum97p5), 
       col = c('blue'), lty = 1, lwd = 1, code = 0)
arrows(x0 = c(0.75,1.75), y0 = c(yrepSum2p5_0, yrepSum2p5), x1 = c(0.75,1.75), y1 = c(yrepSum97p5_0, yrepSum97p5), 
       col = c('red'), lty = 1, lwd = 1, code = 0)
legend('topright', legend=c('y', 'yrep', 'certain y', 'uncertain y'), col = c('blue', 'red', 'black', 'gray'), 
       pch = c(NA,NA,15,15), lty = c(1,1,NA,NA))
dev.off()

#Save data for plotting with premade python functions
write.csv(ppc$y, file = 'DREAMzs_L15_SmithChipVermPrecipPCA_Uncertainty_1915-2020/y.csv', row.names = FALSE)
write.csv(ppc$yrep, file = 'DREAMzs_L15_SmithChipVermPrecipPCA_Uncertainty_1915-2020/yrep.csv', row.names = FALSE)
write.csv(ppc$pi, file = 'DREAMzs_L15_SmithChipVermPrecipPCA_Uncertainty_1915-2020/p.csv', row.names = FALSE)
write.csv(ppc$qi, file = 'DREAMzs_L15_SmithChipVermPrecipPCA_Uncertainty_1915-2020/q.csv', row.names = FALSE)
write.csv(years_L15sm, file = 'DREAMzs_L15_SmithChipVermPrecipPCA_Uncertainty_1915-2020/years.csv', row.names = FALSE)

#Sample from posterior, Lamontagne et al. model----
ppSample_BestLamonMod = function(BTout, #output from a BayesianTools MCMC 
                                 n = 1000,     #number of posterior samples to draw from each chain
                                 Xp,      #the X data to use in regression 
                                 seed,     #the random seed to use
                                 start1915 = FALSE, #Does Xp data start in 1915?
                                 years = NULL #years for 1915-2020 model
){
  #Ensure n is less than length of chains
  if (n > nrow(BTout$chain[[1]])){
    stop('n must be less than or equal to the number of samples in each chain')
  }
  
  set.seed(seed)
  
  if(start1915){
    #Transform Xp with standardization
    MeanX = as.numeric(colMeans(Xp[years > 1961,]))
    SdX = as.numeric(apply(X = Xp[years > 1961,], MARGIN = 2, FUN = sd))
    
    #Fill in matrix
    for (i in 1:ncol(Xp)){
      Xp[,i] = (Xp[,i] - MeanX[i])/SdX[i]
    }
  }
  
  #Get the last n samples from each chain
  param = matrix(NA, nrow = n*length(BTout$chain), ncol = BTout$setup$numPars)
  for (i in 1:length(BTout$chain)){
    param[(1+(n*(i-1))):(n*i),] = BTout$chain[[i]][(nrow(BTout$chain[[i]])-(n-1)):nrow(BTout$chain[[i]]), 1:BTout$setup$numPars]
  }
  
  #Get predicted y (log odds) for each of the parameter sets
  #numerator
  num = t(apply(X = param, MARGIN = 1, FUN = function(p){exp(p[1] + p[2]*Xp[,1] + p[3]*Xp[,2])} ))
  #predicted probability of large IJF
  ppi = t(apply(X = num, MARGIN = 1, FUN = function(num){num/(1+num)}))
  
  #Sample a vector of yrep based on the regression coefficients
  yrep = matrix(NA, nrow = nrow(ppi), ncol = ncol(ppi))
  for (i in 1:ncol(yrep)){
    yrep[,i] = rbinom(n = nrow(yrep), size = 1, prob = ppi[,i])
  }
  
  #return list of params, y and yrep
  retl = list(param = param, pi = ppi, yrep = yrep)
  return(retl)
}

#Posterior Predictive Check for best model
ppc_BestLamonMod = ppSample_BestLamonMod(BTout = outDREAMzs_1p, n = 143, Xp = X_hold[,c(3,5)], seed = 35)
ppc_BestLamonMod_LgSample = ppSample_BestLamonMod(BTout = outDREAMzs_1p, n = 300, Xp = X_hold[,c(3,5)], seed = 36)
bayesplot::ppc_bars(y = Y_hold, yrep = ppc_BestLamonMod$yrep)
#Need to make predictions only for the years with Ft. Smith data 1915-2020 for hindcast.
ppc_BestLamonMod_1915 = ppSample_BestLamonMod(BTout = outDREAMzs_1p, n = 143, Xp = X[-which(is.na(X$Fort.Smith)),c(2,4)], 
                                              seed = 35, start1915 = TRUE, years = years_L15sm)

write.csv(ppc_BestLamonMod_1915$yrep, file = 'DREAMzs_L15_VermPrecip_1962-2020/yrep.csv', row.names = FALSE)
write.csv(ppc_BestLamonMod_1915$pi, file = 'DREAMzs_L15_VermPrecip_1962-2020/p.csv', row.names = FALSE)

#Save betas
write.csv(ppc_BestLamonMod$param, file = 'DREAMzs_L15_VermPrecip_1962-2020/BayesBetas_hold.csv', row.names = FALSE)
write.csv(ppc_BestLamonMod_LgSample$param, file = 'DREAMzs_L15_VermPrecip_1962-2020/BayesBetasLg_hold.csv', row.names = FALSE)

#Sample from posterior, DREAMzs_L15_SmithChipVermPrecipPCA_1962-2020----
ppSample_3p = function(BTout, #output from a BayesianTools MCMC 
                       n = 1000,     #number of posterior samples to draw from each chain
                       Xp,      #the X data to use in regression 
                       seed,     #the random seed to use
                       start1915 = FALSE, #Does Xp data start in 1915?
                       years = NULL, #years for 1915-2020 model
                       PCA = NULL
){
  #Ensure n is less than length of chains
  if (n > nrow(BTout$chain[[1]])){
    stop('n must be less than or equal to the number of samples in each chain')
  }
  
  set.seed(seed)
  
  if(start1915){
    #Transform Xp with standardization
    MeanX = as.numeric(colMeans(Xp[years > 1961,]))
    SdX = as.numeric(apply(X = Xp[years > 1961,], MARGIN = 2, FUN = sd))
    
    #Fill in matrix
    for (i in 1:ncol(Xp)){
      Xp[,i] = (Xp[,i] - MeanX[i])/SdX[i]
    }
    
    if(!is.null(PCA)){
      Xp = predict(PCA, Xp)
    }
  }
  
  #Get the last n samples from each chain
  param = matrix(NA, nrow = n*length(BTout$chain), ncol = BTout$setup$numPars)
  for (i in 1:length(BTout$chain)){
    param[(1+(n*(i-1))):(n*i),] = BTout$chain[[i]][(nrow(BTout$chain[[i]])-(n-1)):nrow(BTout$chain[[i]]), 1:BTout$setup$numPars]
  }
  
  #Get predicted y (log odds) for each of the parameter sets
  #numerator
  num = t(apply(X = param, MARGIN = 1, FUN = function(p){exp(p[1] + p[2]*Xp[,1] + p[3]*Xp[,2])} ))
  #predicted probability of large IJF
  ppi = t(apply(X = num, MARGIN = 1, FUN = function(num){num/(1+num)}))
  
  #Sample a vector of yrep based on the regression coefficients
  yrep = matrix(NA, nrow = nrow(ppi), ncol = ncol(ppi))
  for (i in 1:ncol(yrep)){
    yrep[,i] = rbinom(n = nrow(yrep), size = 1, prob = ppi[,i])
  }
  
  #return list of params, y and yrep
  retl = list(param = param, pi = ppi, yrep = yrep)
  return(retl)
}

#Posterior Predictive Check for best model
ppc_3p = ppSample_3p(BTout = outDREAMzs_3p_AGU, n = 143, Xp = predict(PCAcvs, X_holdsm), seed = 35)
#Need to make predictions only for the years with Ft. Smith data 1915-2020 for hindcast.
ppc_3p_1915 = ppSample_3p(BTout = outDREAMzs_3p_AGU, n = 143, Xp = X[-which(is.na(X$Fort.Smith)),], 
                                              seed = 35, start1915 = TRUE, years = years_L15sm, PCA = PCAcvs)

write.csv(ppc_3p_1915$yrep, file = 'DREAMzs_L15_SmithChipVermPrecipPCA_1962-2020/yrep.csv', row.names = FALSE)
write.csv(ppc_3p_1915$pi, file = 'DREAMzs_L15_SmithChipVermPrecipPCA_1962-2020/p.csv', row.names = FALSE)


#Sample from posterior, no uncertainty 1915-2020----
ppSample_NoUncertainty = function(BTout, #output from a BayesianTools MCMC 
                       n = 1000,     #number of posterior samples to draw from each chain
                       Xp,      #the X data to use in regression 
                       seed     #the random seed to use
){
  #Ensure n is less than length of chains
  if (n > nrow(BTout$chain[[1]])){
    stop('n must be less than or equal to the number of samples in each chain')
  }
  
  set.seed(seed)
  
  #Get the last n samples from each chain
  param = matrix(NA, nrow = n*length(BTout$chain), ncol = BTout$setup$numPars)
  for (i in 1:length(BTout$chain)){
    param[(1+(n*(i-1))):(n*i),] = BTout$chain[[i]][(nrow(BTout$chain[[i]])-(n-1)):nrow(BTout$chain[[i]]), 1:BTout$setup$numPars]
  }
  
  #Get predicted y (log odds) for each of the parameter sets
  #numerator
  num = t(apply(X = param, MARGIN = 1, FUN = function(p){exp(p[1] + p[2]*Xp[,1] + p[3]*Xp[,2])} ))
  #predicted probability of large IJF
  ppi = t(apply(X = num, MARGIN = 1, FUN = function(num){num/(1+num)}))
  
  #Sample a vector of yrep based on the regression coefficients
  yrep = matrix(NA, nrow = nrow(ppi), ncol = ncol(ppi))
  for (i in 1:ncol(yrep)){
    yrep[,i] = rbinom(n = nrow(yrep), size = 1, prob = ppi[,i])
  }
  
  #return list of params, y and yrep
  retl = list(param = param, pi = ppi, yrep = yrep)
  return(retl)
}

#Posterior Predictive Check for best model
ppc_NoUncertainty = ppSample_NoUncertainty(BTout = outDREAMzs_NoUncertainty, n = 143, 
                                           Xp = predict(PCAcvs, X_L15sm), seed = 35)

write.csv(ppc_NoUncertainty$yrep, file = 'DREAMzs_L15_SmithChipVermPrecipPCA_NoUncertainty_1915-2020/yrep.csv', row.names = FALSE)
write.csv(ppc_NoUncertainty$pi, file = 'DREAMzs_L15_SmithChipVermPrecipPCA_NoUncertainty_1915-2020/p.csv', row.names = FALSE)


#Load in GCM predicted temperature and precipitation----
GCMhead = read.csv(file = 'GCM_Temp_Chip.csv', stringsAsFactors = FALSE, nrows=1, header = FALSE)
DDFChip = read.csv(file = 'GCM_Temp_Chip.csv', stringsAsFactors = FALSE, skip = 47, header = FALSE)
DDFVerm = read.csv(file = 'GCM_Temp_Verm.csv', stringsAsFactors = FALSE, skip = 47, header = FALSE)
DDFSmith = read.csv(file = 'GCM_Temp_Smith.csv', stringsAsFactors = FALSE, skip = 47, header = FALSE)
PrecipGPBL = read.csv(file = 'GCM_Precip.csv', stringsAsFactors = FALSE, skip = 47, header = FALSE)

colnames(DDFChip) = colnames(DDFVerm) = colnames(DDFSmith) = colnames(PrecipGPBL) = GCMhead[1,]

#Remove 2100. NA for a few forts
DDFChip = DDFChip[DDFChip$year != 2100,]
#Convert Precip Data
PrecipGPBL[,-1] = PrecipGPBL[,-1]/150.6115385
#Remove years before 1962
DDFChip = DDFChip[DDFChip$year >= 1962,]
DDFVerm = DDFVerm[DDFVerm$year >= 1962,]
DDFSmith = DDFSmith[DDFSmith$year >= 1962,]
PrecipGPBL = PrecipGPBL[PrecipGPBL$year >= 1962,]
#Remove years that cannot be predicted because of no data in historical record
DDFChip_hold = DDFChip[-which(!(DDFChip$year %in% years_hold) & (DDFChip$year <= 2020)),]
DDFVerm_hold = DDFVerm[-which(!(DDFVerm$year %in% years_hold) & (DDFVerm$year <= 2020)),]
DDFSmith_hold = DDFSmith[-which(!(DDFSmith$year %in% years_hold) & (DDFSmith$year <= 2020)),]
PrecipGPBL_hold = PrecipGPBL[-which(!(PrecipGPBL$year %in% years_hold) & (PrecipGPBL$year <= 2020)),]

DDFChip = DDFChip[-which(!(DDFChip$year %in% years_L15sm) & (DDFChip$year <= 2020)),]
DDFVerm = DDFVerm[-which(!(DDFVerm$year %in% years_L15sm) & (DDFVerm$year <= 2020)),]
DDFSmith = DDFSmith[-which(!(DDFSmith$year %in% years_L15sm) & (DDFSmith$year <= 2020)),]
PrecipGPBL = PrecipGPBL[-which(!(PrecipGPBL$year %in% years_L15sm) & (PrecipGPBL$year <= 2020)),]

#Names of GCMs and RCPs
names_GCMs = c(paste(c('HadGEM2-ES','ACCESS1-0','CanESM2','CCSM4','CNRM-CM5','MPI-ESM-LR'), 'RCP85', sep = '_'),
               paste(c('HadGEM2-ES','ACCESS1-0','CanESM2','CCSM4','CNRM-CM5','MPI-ESM-LR'), 'RCP45', sep = '_'))

#GCM probability
dir.create('EDA/ProjectedPCsGCMs', showWarnings = FALSE)
dir.create('EDA/ProjectedPCsGCMs/RawAxes', showWarnings = FALSE)
dir.create('EDA/ProjectedPCsGCMs/PCAxes', showWarnings = FALSE)
GCMprob = function(DDFChip, DDFVerm, DDFSmith, PrecipGPBL, X, ppc, years_L15sm){
  #Transform with standardization and PCA loadings
  MeanGCM = as.numeric(colMeans(X[!apply(X = is.na(as.matrix(X[,c(1,2,3,4)])), MARGIN = 1, FUN = any),]))
  SdGCM = as.numeric(apply(X = X[!apply(X = is.na(as.matrix(X[,c(1,2,3,4)])), MARGIN = 1, FUN = any),], 
                           MARGIN = 2, FUN = sd))
  
  #Array to store probilities. years x GCMs x number of replicates
  pmat = array(data = NA, dim = c(nrow(DDFChip), ncol(DDFChip)-1, nrow(ppc$param)))
  
  #Fill in matrix
  for (i in 1:(ncol(DDFChip)-1)){
    #Matrix of variables for PCA
    mati = cbind((DDFChip[,i+1] - MeanGCM[1])/SdGCM[1], 
                 (DDFVerm[,i+1] - MeanGCM[2])/SdGCM[2], 
                 (DDFSmith[,i+1] - MeanGCM[3])/SdGCM[3], 
                 (PrecipGPBL[,i+1] - MeanGCM[4])/SdGCM[4])
    
    colnames(mati) = colnames(X)[1:4]
    #Get PCs with same transformation as before, and retain first 2
    PCi = predict(object = PCAcvs, newdata = mati)[,1:2]
    
    png(paste0('EDA/ProjectedPCsGCMs/PCAxes/GCM_',names_GCMs[i], '_PCs.png'), res = 300, units = 'in', width = 5, height = 5)
    plot(predPCAcvs[,1], predPCAcvs[,2], ylim = c(-3,6), xlim = c(-10,5), pch = 16, col = 'gray',
         xlab = 'PC1', ylab = 'PC2', main = names_GCMs[i], cex.axis = 0.5, cex.lab = 0.9)
    par(new = T)
    plot(predPCAcvs[Y_L15sm == 1,1], predPCAcvs[Y_L15sm == 1,2], ylim = c(-3,6), 
         xlim = c(-10,5), pch = 16, col = 'orange', axes = FALSE, xlab = '', ylab = '')
    par(new = T)
    plot(PCi[,1], PCi[,2], ylim = c(-3,6), xlim = c(-10,5), pch = 16, axes = FALSE, xlab = '', ylab = '')
    legend('bottomleft', legend = c('Projected Conditions', '1915-2020 Conditions', 'Recorded Large Floods'),
           col = c('black', 'gray', 'orange'), pch = 16, cex = 0.7)
    dev.off()
    
    #Use PCs to predict IJFs with sampled params matrix
    #Get predicted p prob for each of the parameter sets
    #numerator
    num = t(apply(X = ppc$param, MARGIN = 1, FUN = function(p){exp(p[1] + p[2]*PCi[,1] + p[3]*PCi[,2])}))
    #predicted probability of large IJF
    ppi = apply(X = num, MARGIN = 1, FUN = function(num){num/(1+num)})
    
    #Probability before 2020 replaced with historical estimates
    ppi[which(DDFChip$year <= 2020), ] = t(ppc$pi[,which(years_L15sm > 1961)])
    
    #Append predicted probs
    pmat[,i,] = ppi
  }
  return(pmat)
}

GCMprob_hold = function(DDFVerm_hold, PrecipGPBL_hold, X, ppc_BestLamonMod, years_hold){
  #Transform with standardization
  MeanGCM = as.numeric(colMeans(X[years>1961,]))
  SdGCM = as.numeric(apply(X = X[years>1961,], MARGIN = 2, FUN = sd))
  
  #Array to store probilities. years x GCMs x number of replicates
  pmat = array(data = NA, dim = c(nrow(DDFVerm_hold), ncol(DDFVerm_hold)-1, nrow(ppc_BestLamonMod$param)))
  
  #Fill in matrix
  for (i in 1:(ncol(DDFVerm_hold)-1)){
    #Matrix of variables for PCA
    mati = cbind((DDFVerm_hold[,i+1] - MeanGCM[2])/SdGCM[2], 
                 (PrecipGPBL_hold[,i+1] - MeanGCM[4])/SdGCM[4])
    
    png(paste0('EDA/ProjectedPCsGCMs/RawAxes/GCM_',names_GCMs[i], '_FtVermGPBL.png'), res = 300, units = 'in', width = 5, height = 5)
    plot(X_hold[,3], X_hold[,5], pch = 16, col = 'gray',
         xlab = 'Ft. Vermillion DDF', ylab = 'GP/BL Precipitation', main = names_GCMs[i], 
         cex.axis = 0.5, cex.lab = 0.9, xlim = c(-5,5), ylim = c(-4,5))
    par(new = T)
    plot(X_hold[Y_hold == 1,3], X_hold[Y_hold == 1,5], 
         xlim = c(-5,5), ylim = c(-4,5), pch = 16, col = 'orange', axes = FALSE, xlab = '', ylab = '')
    par(new = T)
    plot(mati[,1], mati[,2], xlim = c(-5,5), ylim = c(-4,5), pch = 16, axes = FALSE, xlab = '', ylab = '')
    legend('bottomleft', legend = c('Projected Conditions', '1915-2020 Conditions', 'Recorded Large Floods'),
           col = c('black', 'gray', 'orange'), pch = 16, cex = 0.7)
    dev.off()
    
    #Predict IJFs with sampled params matrix
    #Get predicted p prob for each of the parameter sets
    #numerator
    num = t(apply(X = ppc_BestLamonMod$param, MARGIN = 1, FUN = function(p){exp(p[1] + p[2]*mati[,1] + p[3]*mati[,2])}))
    #predicted probability of large IJF
    ppi = apply(X = num, MARGIN = 1, FUN = function(num){num/(1+num)})
    
    #Probability before 2020 replaced with historical estimates
    ppi[which(DDFVerm_hold$year<=2020), ] = t(ppc_BestLamonMod$pi)
    
    #Append predicted probs
    pmat[,i,] = ppi
  }
  return(pmat)
}

pmatGCM = GCMprob(DDFChip, DDFVerm, DDFSmith, PrecipGPBL, X, ppc, years_L15sm)
pmatGCM_hold = GCMprob_hold(DDFVerm_hold = DDFVerm_hold, PrecipGPBL_hold = PrecipGPBL_hold, X = X, 
                            ppc_BestLamonMod = ppc_BestLamonMod, years_hold = years_hold)
pmatGCM_3p = GCMprob(DDFChip, DDFVerm, DDFSmith, PrecipGPBL, X, ppc_3p_1915, years_L15sm)
pmatGCM_NoUncertainty = GCMprob(DDFChip, DDFVerm, DDFSmith, PrecipGPBL, X, ppc_NoUncertainty, years_L15sm)

#Save for loading into plotting functions in Python
write.csv(pmatGCM[,1,], file = 'DREAMzs_L15_SmithChipVermPrecipPCA_Uncertainty_1915-2020/GCMp_Had85.csv', row.names = FALSE)
write.csv(pmatGCM[,2,], file = 'DREAMzs_L15_SmithChipVermPrecipPCA_Uncertainty_1915-2020/GCMp_Acc85.csv', row.names = FALSE)
write.csv(pmatGCM[,3,], file = 'DREAMzs_L15_SmithChipVermPrecipPCA_Uncertainty_1915-2020/GCMp_Can85.csv', row.names = FALSE)
write.csv(pmatGCM[,4,], file = 'DREAMzs_L15_SmithChipVermPrecipPCA_Uncertainty_1915-2020/GCMp_CCS85.csv', row.names = FALSE)
write.csv(pmatGCM[,5,], file = 'DREAMzs_L15_SmithChipVermPrecipPCA_Uncertainty_1915-2020/GCMp_CNR85.csv', row.names = FALSE)
write.csv(pmatGCM[,6,], file = 'DREAMzs_L15_SmithChipVermPrecipPCA_Uncertainty_1915-2020/GCMp_MPI85.csv', row.names = FALSE)
write.csv(pmatGCM[,7,], file = 'DREAMzs_L15_SmithChipVermPrecipPCA_Uncertainty_1915-2020/GCMp_Had45.csv', row.names = FALSE)
write.csv(pmatGCM[,8,], file = 'DREAMzs_L15_SmithChipVermPrecipPCA_Uncertainty_1915-2020/GCMp_Acc45.csv', row.names = FALSE)
write.csv(pmatGCM[,9,], file = 'DREAMzs_L15_SmithChipVermPrecipPCA_Uncertainty_1915-2020/GCMp_Can45.csv', row.names = FALSE)
write.csv(pmatGCM[,10,], file = 'DREAMzs_L15_SmithChipVermPrecipPCA_Uncertainty_1915-2020/GCMp_CCS45.csv', row.names = FALSE)
write.csv(pmatGCM[,11,], file = 'DREAMzs_L15_SmithChipVermPrecipPCA_Uncertainty_1915-2020/GCMp_CNR45.csv', row.names = FALSE)
write.csv(pmatGCM[,12,], file = 'DREAMzs_L15_SmithChipVermPrecipPCA_Uncertainty_1915-2020/GCMp_MPI45.csv', row.names = FALSE)

write.csv(pmatGCM_hold[,1,], file = 'DREAMzs_L15_VermPrecip_1962-2020/GCMp_Had85_hold.csv', row.names = FALSE)
write.csv(pmatGCM_hold[,2,], file = 'DREAMzs_L15_VermPrecip_1962-2020/GCMp_Acc85_hold.csv', row.names = FALSE)
write.csv(pmatGCM_hold[,3,], file = 'DREAMzs_L15_VermPrecip_1962-2020/GCMp_Can85_hold.csv', row.names = FALSE)
write.csv(pmatGCM_hold[,4,], file = 'DREAMzs_L15_VermPrecip_1962-2020/GCMp_CCS85_hold.csv', row.names = FALSE)
write.csv(pmatGCM_hold[,5,], file = 'DREAMzs_L15_VermPrecip_1962-2020/GCMp_CNR85_hold.csv', row.names = FALSE)
write.csv(pmatGCM_hold[,6,], file = 'DREAMzs_L15_VermPrecip_1962-2020/GCMp_MPI85_hold.csv', row.names = FALSE)
write.csv(pmatGCM_hold[,7,], file = 'DREAMzs_L15_VermPrecip_1962-2020/GCMp_Had45_hold.csv', row.names = FALSE)
write.csv(pmatGCM_hold[,8,], file = 'DREAMzs_L15_VermPrecip_1962-2020/GCMp_Acc45_hold.csv', row.names = FALSE)
write.csv(pmatGCM_hold[,9,], file = 'DREAMzs_L15_VermPrecip_1962-2020/GCMp_Can45_hold.csv', row.names = FALSE)
write.csv(pmatGCM_hold[,10,], file = 'DREAMzs_L15_VermPrecip_1962-2020/GCMp_CCS45_hold.csv', row.names = FALSE)
write.csv(pmatGCM_hold[,11,], file = 'DREAMzs_L15_VermPrecip_1962-2020/GCMp_CNR45_hold.csv', row.names = FALSE)
write.csv(pmatGCM_hold[,12,], file = 'DREAMzs_L15_VermPrecip_1962-2020/GCMp_MPI45_hold.csv', row.names = FALSE)

write.csv(pmatGCM_3p[,1,], file = 'DREAMzs_L15_SmithChipVermPrecipPCA_1962-2020/GCMp_Had85_cvsp.csv', row.names = FALSE)
write.csv(pmatGCM_3p[,2,], file = 'DREAMzs_L15_SmithChipVermPrecipPCA_1962-2020/GCMp_Acc85_cvsp.csv', row.names = FALSE)
write.csv(pmatGCM_3p[,3,], file = 'DREAMzs_L15_SmithChipVermPrecipPCA_1962-2020/GCMp_Can85_cvsp.csv', row.names = FALSE)
write.csv(pmatGCM_3p[,4,], file = 'DREAMzs_L15_SmithChipVermPrecipPCA_1962-2020/GCMp_CCS85_cvsp.csv', row.names = FALSE)
write.csv(pmatGCM_3p[,5,], file = 'DREAMzs_L15_SmithChipVermPrecipPCA_1962-2020/GCMp_CNR85_cvsp.csv', row.names = FALSE)
write.csv(pmatGCM_3p[,6,], file = 'DREAMzs_L15_SmithChipVermPrecipPCA_1962-2020/GCMp_MPI85_cvsp.csv', row.names = FALSE)
write.csv(pmatGCM_3p[,7,], file = 'DREAMzs_L15_SmithChipVermPrecipPCA_1962-2020/GCMp_Had45_cvsp.csv', row.names = FALSE)
write.csv(pmatGCM_3p[,8,], file = 'DREAMzs_L15_SmithChipVermPrecipPCA_1962-2020/GCMp_Acc45_cvsp.csv', row.names = FALSE)
write.csv(pmatGCM_3p[,9,], file = 'DREAMzs_L15_SmithChipVermPrecipPCA_1962-2020/GCMp_Can45_cvsp.csv', row.names = FALSE)
write.csv(pmatGCM_3p[,10,], file = 'DREAMzs_L15_SmithChipVermPrecipPCA_1962-2020/GCMp_CCS45_cvsp.csv', row.names = FALSE)
write.csv(pmatGCM_3p[,11,], file = 'DREAMzs_L15_SmithChipVermPrecipPCA_1962-2020/GCMp_CNR45_cvsp.csv', row.names = FALSE)
write.csv(pmatGCM_3p[,12,], file = 'DREAMzs_L15_SmithChipVermPrecipPCA_1962-2020/GCMp_MPI45_cvsp.csv', row.names = FALSE)

write.csv(pmatGCM_NoUncertainty[,1,], file = 'DREAMzs_L15_SmithChipVermPrecipPCA_NoUncertainty_1915-2020/GCMp_Had85_NoUncertainty.csv', row.names = FALSE)
write.csv(pmatGCM_NoUncertainty[,2,], file = 'DREAMzs_L15_SmithChipVermPrecipPCA_NoUncertainty_1915-2020/GCMp_Acc85_NoUncertainty.csv', row.names = FALSE)
write.csv(pmatGCM_NoUncertainty[,3,], file = 'DREAMzs_L15_SmithChipVermPrecipPCA_NoUncertainty_1915-2020/GCMp_Can85_NoUncertainty.csv', row.names = FALSE)
write.csv(pmatGCM_NoUncertainty[,4,], file = 'DREAMzs_L15_SmithChipVermPrecipPCA_NoUncertainty_1915-2020/GCMp_CCS85_NoUncertainty.csv', row.names = FALSE)
write.csv(pmatGCM_NoUncertainty[,5,], file = 'DREAMzs_L15_SmithChipVermPrecipPCA_NoUncertainty_1915-2020/GCMp_CNR85_NoUncertainty.csv', row.names = FALSE)
write.csv(pmatGCM_NoUncertainty[,6,], file = 'DREAMzs_L15_SmithChipVermPrecipPCA_NoUncertainty_1915-2020/GCMp_MPI85_NoUncertainty.csv', row.names = FALSE)
write.csv(pmatGCM_NoUncertainty[,7,], file = 'DREAMzs_L15_SmithChipVermPrecipPCA_NoUncertainty_1915-2020/GCMp_Had45_NoUncertainty.csv', row.names = FALSE)
write.csv(pmatGCM_NoUncertainty[,8,], file = 'DREAMzs_L15_SmithChipVermPrecipPCA_NoUncertainty_1915-2020/GCMp_Acc45_NoUncertainty.csv', row.names = FALSE)
write.csv(pmatGCM_NoUncertainty[,9,], file = 'DREAMzs_L15_SmithChipVermPrecipPCA_NoUncertainty_1915-2020/GCMp_Can45_NoUncertainty.csv', row.names = FALSE)
write.csv(pmatGCM_NoUncertainty[,10,], file = 'DREAMzs_L15_SmithChipVermPrecipPCA_NoUncertainty_1915-2020/GCMp_CCS45_NoUncertainty.csv', row.names = FALSE)
write.csv(pmatGCM_NoUncertainty[,11,], file = 'DREAMzs_L15_SmithChipVermPrecipPCA_NoUncertainty_1915-2020/GCMp_CNR45_NoUncertainty.csv', row.names = FALSE)
write.csv(pmatGCM_NoUncertainty[,12,], file = 'DREAMzs_L15_SmithChipVermPrecipPCA_NoUncertainty_1915-2020/GCMp_MPI45_NoUncertainty.csv', row.names = FALSE)

#Save all floods for use in plotting
Y_L62t20 = Data$Flood[Data$Year >= 1962]
write.csv(Y_L62t20, file = 'DREAMzs_L15_SmithChipVermPrecipPCA_Uncertainty_1915-2020/yL62t20.csv', row.names = FALSE)
write.csv(Y_L15sm[years_L15sm > 1961], file = 'DREAMzs_L15_SmithChipVermPrecipPCA_Uncertainty_1915-2020/yL15sm62t20.csv', 
          row.names = FALSE)
write.csv(Y_hold[years_hold>1961], file = 'DREAMzs_L15_VermPrecip_1962-2020/y62t20.csv', row.names = FALSE)
#Save years
write.csv(DDFChip$year, file = 'DREAMzs_L15_SmithChipVermPrecipPCA_Uncertainty_1915-2020/year62t2099.csv', row.names = FALSE)
write.csv(DDFChip_hold$year, file = 'DREAMzs_L15_VermPrecip_1962-2020/year62t2099.csv', row.names = FALSE)

#Diagnose GCMs----
#Vermillion consistently warmer than others. Smith and Chip most similar
for (i in 1:(ncol(DDFSmith)-1)){
  plot(DDFSmith$year[DDFSmith$year>2020], DDFSmith[DDFSmith$year>2020,i+1], type = 'l', ylim = c(-4000,1000))
  par(new=T)
  plot(DDFChip$year[DDFChip$year>2020], DDFChip[DDFChip$year>2020,i+1], type = 'l', ylim = c(-4000,1000), col = 'blue')
  par(new=T)
  plot(DDFVerm$year[DDFVerm$year>2020], DDFVerm[DDFSmith$year>2020,i+1], type = 'l', ylim = c(-4000,1000), col = 'red')
}
for (i in 1:(ncol(PrecipGPBL)-1)){
  plot(PrecipGPBL$year, PrecipGPBL[,i+1], type = 'l', ylim = c(0,3))
}

#No outliers on scatterplot matrices
for (i in 1:(ncol(PrecipGPBL)-1)){
  plot(as.data.frame(list(Smith = DDFSmith[DDFSmith$year>2020,i+1], Chip = DDFChip[DDFChip$year>2020,i+1], Verm = DDFVerm[DDFVerm$year>2020,i+1])))
}