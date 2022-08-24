#Edit BayesianTools WAIC function to return correct values
WAIC = function (bayesianOutput, numSamples = 1000, ...) 
{
  x = getSample(bayesianOutput, parametersOnly = F, ...)
  if ("mcmcSamplerList" %in% class(bayesianOutput)) {
    if (bayesianOutput[[1]]$setup$pwLikelihood == FALSE) 
      stop("WAIC can only be applied if the likelihood density can be returned point-wise ('sum' argument, see examples).")
    nPars = bayesianOutput[[1]]$setup$numPars
    llDensity <- bayesianOutput[[1]]$setup$likelihood$density
  }
  else {
    if (bayesianOutput$setup$pwLikelihood == FALSE) 
      stop("WAIC can only be applied if the likelihood density can be returned point-wise ('sum' argument, see examples).")
    nPars = bayesianOutput$setup$numPars
    llDensity <- bayesianOutput$setup$likelihood$density
  }
  i <- sample.int(nrow(x), numSamples, replace = TRUE)
  #removed the t() around llDensity.
  pointWiseLikelihood = llDensity(x[i, 1:nPars], sum = F)
  lppd <- sum(apply(pointWiseLikelihood, 2, logSumExp, mean = T))
  pWAIC1 <- 2 * sum(apply(pointWiseLikelihood, 2, function(y) logSumExp(y, 
                                                                        mean = T) - mean(y)))
  pWAIC2 <- sum(apply(pointWiseLikelihood, 2, var))
  out = list(WAIC1 = -2 * (lppd - pWAIC1), WAIC2 = -2 * (lppd - 
                                                           pWAIC2), lppd = lppd, pWAIC1 = pWAIC1, pWAIC2 = pWAIC2)
  return(out)
}
logSumExp<- function(x, mean = F) {
  # 
  
  nObs = length(x)   
  
  if(any(x == Inf)) stop("BayesianTools::logSumExp: positive infinity values in log probabilities")
  if(any(x == -Inf )){
    message("BayesianTools::logSumExp: encountered -Inf in logSumExp - value was removed")    
    x = x[x != -Inf] 
  } 
  
  # seems that this created problems in the presence of small values,
  # doesn't seem to be a need to shift towards min
  # if ( max(abs(x)) > max(x) ) offset <- min(x) else offset <- max(x)
  offset <- max(x)
  if (mean == T) out = log(sum(exp(x - offset))/nObs) + offset
  else out = log(sum(exp(x - offset))) + offset
  return(out)
}
#Second order WAIC - BayesianTools
WAIC2 = function (bayesianOutput, numSamples = 1000, ...) 
{
  x = getSample(bayesianOutput, parametersOnly = F, ...)
  if ("mcmcSamplerList" %in% class(bayesianOutput)) {
    if (bayesianOutput[[1]]$setup$pwLikelihood == FALSE) 
      stop("WAIC can only be applied if the likelihood density can be returned point-wise ('sum' argument, see examples).")
    nPars = bayesianOutput[[1]]$setup$numPars
    llDensity <- bayesianOutput[[1]]$setup$likelihood$density
  }
  else {
    if (bayesianOutput$setup$pwLikelihood == FALSE) 
      stop("WAIC can only be applied if the likelihood density can be returned point-wise ('sum' argument, see examples).")
    nPars = bayesianOutput$setup$numPars
    llDensity <- bayesianOutput$setup$likelihood$density
  }
  i <- sample.int(nrow(x), numSamples, replace = TRUE)
  pointWiseLikelihood = llDensity(x[i, 1:nPars], sum = F)
  lppd <- sum(apply(pointWiseLikelihood, 2, logSumExp, mean = T))
  pWAIC1 <- 2 * sum(apply(pointWiseLikelihood, 2, function(y) logSumExp(y, 
                                                                        mean = T) - mean(y)))
  pWAIC2 <- sum(apply(pointWiseLikelihood, 2, var))
  out = list(WAIC1 = -2 * (lppd - pWAIC1) + (2*(pWAIC1)^2 + 2*(pWAIC1))/(ncol(pointWiseLikelihood) - pWAIC1 - 1), 
             WAIC2 = -2 * (lppd - pWAIC2) + (2*(pWAIC2)^2 + 2*(pWAIC2))/(ncol(pointWiseLikelihood) - pWAIC2 - 1), 
             lppd = lppd, pWAIC1 = pWAIC1, pWAIC2 = pWAIC2)
  return(out)
}


#Second order WAIC - loo
#Not returning all info. Would have to update with package compilation to do that
waic2 = function (x, ...) 
{
  ll <- validate_ll(x)
  lldim <- dim(ll)
  lpd <- colLogMeanExps(ll)
  p_waic <- matrixStats::colVars(ll)
  elpd_waic <- lpd - p_waic
  waic <- sum(-2 * elpd_waic) + (2*(sum(p_waic))^2 + 2*(sum(p_waic)))/(lldim[2] - sum(p_waic) - 1)
  pointwise <- cbind(elpd_waic, p_waic, waic)
  #throw_pwaic_warnings(pointwise[, "p_waic"], digits = 1)
  return(waic)
}
validate_ll <- function(x) {
  if (is.list(x)) {
    stop("List not allowed as input.")
  } else if (anyNA(x)) {
    stop("NAs not allowed in input.")
  } else if (!all(is.finite(x))) {
    stop("All input values must be finite.")
  }
  invisible(x)
}
colLogMeanExps <- function(x) {
  logS <- log(nrow(x))
  matrixStats::colLogSumExps(x) - logS
}
