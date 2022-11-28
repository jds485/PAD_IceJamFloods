#Normalize X data and add a constant column for regression
#X is a matrix with years as rows and features as columns
normalize = function(X){
  #Normalize for all columns
  k = dim(X)[2]
  for (i in 1:k){
    X[, i] = (X[, i] - mean(X[, i]))/sd(X[, i])
  }
  
  #add a constant
  X = cbind(rep(1.0, nrow(X)), X)
  colnames(X)[1] = 'Constant'
  
  return(X)
}

#Corrected (second order) and regular AIC
#FirthMod is the output from the logistf function
#it must have columns for:
#n - number of observations
#loglik - log likelihood
#df - degrees of freedom
aicc = function(FirthMod){
  #degrees of freedom
  df = FirthMod$df + 1
  
  #aicc
  a = -2*FirthMod$loglik[2] + 2*df + (2*(df)^2 + 2*df)/(FirthMod$n - df - 1)
  
  return(a)
}
aic = function(FirthMod){
  a = -2*FirthMod$loglik[2] + 2*(FirthMod$df+1)
  
  return(a)
}

#Removes MCMC chain steps that have Inf likelihoods in the first chain step
#This occurs when the initial betas are very unrealistic
#bayesianOutput is the output from BayesianTools::runMCMC
removeInf = function(bayesianOutput){
  #Log likelihood column index
  LLind = bayesianOutput$setup$numPars + 1
  
  #Infinity check vector - one element per MCMC chain
  InfCk = vector('numeric', length(bayesianOutput$chain))
  
  for (i in 1:length(InfCk)){
    if (is.infinite(bayesianOutput$chain[[i]][1, LLind])){
      #Replace first step in chain with the second
      bayesianOutput$chain[[i]][1,] = bayesianOutput$chain[[i]][2,]
    }
    #Check there are no other infinities
    InfCk[i] = length(which(is.infinite(bayesianOutput$chain[[i]])))
  }
  bayesianOutput$InfCk = InfCk
  return(bayesianOutput)
}


#Figures
plot_EDA_AllFloods = function(fname, X, X_colname, X_label, Y, FloodMag){
  #Y is GP/BL precip, X is a temperature variable
  png(fname, res = 300, units = 'in', width = 5, height = 5)
  plot(x = X[[X_colname]], y = X$GP.BL.Precip.pct.Avg, pch = 16, ylim = c(-2,4), xlim = c(-3,3), 
       ylab='GP/BL Nov.-Apr. Precip.', xlab = X_label, cex.axis = 1.5, cex.lab = 1.5)
  par(new = TRUE)
  plot(x = X[[X_colname]][which((Y == 1) & (FloodMag == 'L'))], 
       y = X$GP.BL.Precip.pct.Avg[which((Y == 1) & (FloodMag == 'L'))], 
       col = 'orange', pch = 16, ylim = c(-2,4), xlim = c(-3,3), ann = FALSE, axes = FALSE)
  par(new = TRUE)
  plot(x = X[[X_colname]][which((Y == 1) & (FloodMag == 'M'))], 
       y = X$GP.BL.Precip.pct.Avg[which((Y == 1) & (FloodMag == 'M'))], 
       col = 'green', pch = 16, ylim = c(-2,4), xlim = c(-3,3), ann = FALSE, axes = FALSE)
  par(new = TRUE)
  plot(x = X[[X_colname]][which((Y == 1) & (FloodMag == 'S'))], 
       y = X$GP.BL.Precip.pct.Avg[which((Y == 1) & (FloodMag == 'S'))], 
       col = 'blue', pch = 16, ylim = c(-2,4), xlim = c(-3,3), ann = FALSE, axes = FALSE)
  par(new = TRUE)
  plot(x = X[[X_colname]][which((FloodMag == 'U'))], 
       y = X$GP.BL.Precip.pct.Avg[which((FloodMag == 'U'))], 
       col = 'gray', pch = 16, ylim = c(-2,4), xlim = c(-3,3), ann = FALSE, axes = FALSE)
  legend('topright', legend = c('Large Flood', 'Moderate Flood', 'Small Flood', 'Unknown Magnitude', 'No Flood'), 
         col = c('orange', 'green', 'blue', 'gray', 'black'), pch = 16, cex = 0.7)
  dev.off()
}

plot_EDA_MSFloodsPre1962 = function(fname, X, X_colname, X_label, Y, FloodMag, years){
  png(fname, res = 300, units = 'in', width = 5, height = 5)
  plot(x = X[[X_colname]], y = X$GP.BL.Precip.pct.Avg, pch = 16, ylim = c(-2,4), xlim = c(-3,3), 
       ylab='GP/BL Nov.-Apr. Precip.', xlab = X_label, cex.axis = 1.5, cex.lab = 1.5)
  par(new = TRUE)
  plot(x = X[[X_colname]][which((Y == 1) & (FloodMag == 'L'))], 
       y = X$GP.BL.Precip.pct.Avg[which((Y == 1) & (FloodMag == 'L'))], 
       col = 'darkorange3', pch = 16, ylim = c(-2,4), xlim = c(-3,3), ann = FALSE, axes = FALSE)
  par(new = TRUE)
  plot(x = X[[X_colname]][which((Y == 1) & (FloodMag == 'L') & (years < 1962))], 
       y = X$GP.BL.Precip.pct.Avg[which((Y == 1) & (FloodMag == 'L') & (years < 1962))], 
       col = 'orange', pch = 16, ylim = c(-2,4), xlim = c(-3,3), ann = FALSE, axes = FALSE)
  par(new = TRUE)
  plot(x = X[[X_colname]][which((Y == 1) & (FloodMag == 'M') & (years < 1962))], 
       y = X$GP.BL.Precip.pct.Avg[which((Y == 1) & (FloodMag == 'M') & (years < 1962))], 
       col = 'green', pch = 16, ylim = c(-2,4), xlim = c(-3,3), ann = FALSE, axes = FALSE)
  par(new = TRUE)
  plot(x = X[[X_colname]][which((Y == 1) & (FloodMag == 'S') & (years < 1962))], 
       y = X$GP.BL.Precip.pct.Avg[which((Y == 1) & (FloodMag == 'S') & (years < 1962))], 
       col = 'blue', pch = 16, ylim = c(-2,4), xlim = c(-3,3), ann = FALSE, axes = FALSE)
  par(new = TRUE)
  plot(x = X[[X_colname]][which((FloodMag == 'U'))], 
       y = X$GP.BL.Precip.pct.Avg[which((FloodMag == 'U'))], 
       col = 'gray', pch = 16, ylim = c(-2,4), xlim = c(-3,3), ann = FALSE, axes = FALSE)
  legend('topright', legend = c('Large Flood (1962-2020)', 'Large Flood (pre-1962)', 'Moderate Flood (pre-1962)', 
                                'Small Flood (pre-1962)', 'Unknown Magnitude', 'No Flood/Not Large (1962-2020)'), 
         col = c('darkorange3', 'orange', 'green', 'blue', 'gray', 'black'), pch = 16, cex = 0.7)
  dev.off()
}

plot_EDA_PCA_MSFloodsPre1962_color = function(fname, X, X_label, Y_label, Y, FloodMag, years){
  #Coordinates of loadings axes
  load_x <- X$rotation[,1]*3
  load_y <- X$rotation[,2]*3
  #Text for axes labels
  # Label position
  text_pos <- load_y
  lo <- which(load_y < 0) # Get the variables on the bottom half of the plot
  hi <- which(load_y >= 0) # Get variables on the top half
  # Replace values in the vector
  text_pos <- replace(text_pos, lo, "1")
  text_pos <- replace(text_pos, hi, "3")
  
  png(fname, res = 300, units = 'in', width = 5, height = 5)
  par(mar = c(4,4,1,1))
  plot(x = X$x[,1], y = X$x[,2], pch = 16, ylim = c(-3,3), xlim = c(-5,5),
       ylab = Y_label, 
       xlab = X_label, 
       cex.lab = 0.9, 
       axes = FALSE)
  axis(side = 1, at = seq(-5,5,1), labels = TRUE, cex.axis = 0.5)
  axis(side = 2, at = seq(-3,3,1), labels = TRUE, cex.axis = 0.5)
  box()
  par(new = TRUE)
  plot(x = X$x[,1][which((Y == 1) & (FloodMag == 'L'))], 
       y = X$x[,2][which((Y == 1) & (FloodMag == 'L'))], 
       col = 'orange3', pch = 16, ylim = c(-3,3), xlim = c(-5,5), ann = FALSE, axes = FALSE)
  par(new = TRUE)
  plot(x = X$x[,1][which((Y == 1) & (FloodMag == 'L') & (years < 1962))], 
       y = X$x[,2][which((Y == 1) & (FloodMag == 'L') & (years < 1962))], 
       col = 'orange', pch = 16, ylim = c(-3,3), xlim = c(-5,5), ann = FALSE, axes = FALSE)
  par(new = TRUE)
  plot(x = X$x[,1][which((Y == 1) & (FloodMag == 'M') & (years < 1962))], 
       y = X$x[,2][which((Y == 1) & (FloodMag == 'M') & (years < 1962))], 
       col = 'green', pch = 16, ylim = c(-3,3), xlim = c(-5,5), ann = FALSE, axes = FALSE)
  par(new = TRUE)
  plot(x = X$x[,1][which((Y == 1) & (FloodMag == 'S') & (years < 1962))], 
       y = X$x[,2][which((Y == 1) & (FloodMag == 'S') & (years < 1962))], 
       col = 'blue', pch = 16, ylim = c(-3,3), xlim = c(-5,5), ann = FALSE, axes = FALSE)
  par(new = TRUE)
  plot(x = X$x[,1][which((FloodMag == 'U'))], 
       y = X$x[,2][which((FloodMag == 'U'))], 
       col = 'hotpink', pch = 16, ylim = c(-3,3), xlim = c(-5,5), ann = FALSE, axes = FALSE)
  par(new = TRUE)
  plot(x = X$x[,1][which((Y == 0) & (FloodMag == 'N') & (years < 1962))], 
       y = X$x[,2][which((Y == 0) & (FloodMag == 'N') & (years < 1962))], 
       col = gray(0.6), pch = 16, ylim = c(-3,3), xlim = c(-5,5), ann = FALSE, axes = FALSE)
  legend('bottomleft', legend = c('Large (1962-2020)', 'Large (pre-1962)', 'Moderate (pre-1962)', 
                                  'Small (pre-1962)', 'Unknown (pre-1962)', 'No Flood or Not Large (1962-2020)',
                                  'No Flood (pre-1962)'),
         title = expression(bold('Flood Magnitude')), title.adj = 0.1,
         col = c('orange3', 'orange', 'green', 'blue', 'hotpink', 'black', gray(0.6)), pch = 16, cex = .7, bty = 'n')
  par(xpd = TRUE)
  #text(x = -6.9, y = 3.2, 'More\nSnow')
  #text(x = 5, y = -4, 'Colder')
  #plot biplot axes for each variable
  arrows(x0 = 0, x1 = load_x, y0 = 0, y1 = load_y, col = "black", length = 0.1, lwd = 1.5)
  text(load_x, load_y, labels = c('','','DDF','Precipitation'), col  ="black", pos = text_pos, cex = 0.7)
  par(xpd = FALSE)
  dev.off()
}

plot_EDA_PCA_MSFloodsPre1962_fill = function(fname, X, X_label, Y_label, Y, FloodMag, years, pdf = FALSE,
                                             label_mag = NULL){
  #Coordinates of loadings axes
  load_x <- X$rotation[,1]*3
  load_y <- X$rotation[,2]*3
  #Text for axes labels
  # Label position
  text_pos <- load_y
  lo <- which(load_y < 0) # Get the variables on the bottom half of the plot
  hi <- which(load_y >= 0) # Get variables on the top half
  # Replace values in the vector
  text_pos <- replace(text_pos, lo, "1")
  text_pos <- replace(text_pos, hi, "3")
  
  if (pdf){
    pdf(fname, width = 5, height = 5)
  }else{
    png(fname, res = 300, units = 'in', width = 5, height = 5)
  }
  par(mar = c(4,4,1,1))
  plot(x = X$x[,1], y = X$x[,2], pch = 16, ylim = c(-3,3), xlim = c(-5,5),
       ylab = Y_label, 
       xlab = X_label, 
       cex.lab = 0.9, 
       col = 'white', 
       axes = FALSE)
  axis(side = 1, at = seq(-5,5,1), labels = TRUE, cex.axis = 0.5)
  axis(side = 2, at = seq(-3,3,1), labels = TRUE, cex.axis = 0.5)
  box()
  par(new = TRUE)
  plot(x = X$x[,1][which((FloodMag == 'N' | FloodMag == 'M' | FloodMag == 'S') & (years >= 1962))], 
       y = X$x[,2][which((FloodMag == 'N' | FloodMag == 'M' | FloodMag == 'S') & (years >= 1962))], 
       col = 'black', pch = 16, ylim = c(-3,3), xlim = c(-5,5), ann = FALSE, axes = FALSE)
  par(new = TRUE)
  plot(x = X$x[,1][which((Y == 1) & (FloodMag == 'L') & (years >= 1962))], 
       y = X$x[,2][which((Y == 1) & (FloodMag == 'L') & (years >= 1962))], 
       col = 'orange', pch = 16, ylim = c(-3,3), xlim = c(-5,5), ann = FALSE, axes = FALSE)
  par(new = TRUE)
  plot(x = X$x[,1][which((Y == 1) & (FloodMag == 'L') & (years < 1962))], 
       y = X$x[,2][which((Y == 1) & (FloodMag == 'L') & (years < 1962))], 
       col = 'orange', pch = 1, ylim = c(-3,3), xlim = c(-5,5), ann = FALSE, axes = FALSE)
  par(new = TRUE)
  plot(x = X$x[,1][which((Y == 1) & (FloodMag == 'M') & (years < 1962))], 
       y = X$x[,2][which((Y == 1) & (FloodMag == 'M') & (years < 1962))], 
       col = 'green', pch = 1, ylim = c(-3,3), xlim = c(-5,5), ann = FALSE, axes = FALSE)
  par(new = TRUE)
  plot(x = X$x[,1][which((Y == 1) & (FloodMag == 'S') & (years < 1962))], 
       y = X$x[,2][which((Y == 1) & (FloodMag == 'S') & (years < 1962))], 
       col = 'blue', pch = 1, ylim = c(-3,3), xlim = c(-5,5), ann = FALSE, axes = FALSE)
  par(new = TRUE)
  plot(x = X$x[,1][which((FloodMag == 'U'))], 
       y = X$x[,2][which((FloodMag == 'U'))], 
       col = 'hotpink', pch = 1, ylim = c(-3,3), xlim = c(-5,5), ann = FALSE, axes = FALSE)
  par(new = TRUE)
  plot(x = X$x[,1][which((Y == 0) & (FloodMag == 'N') & (years < 1962))], 
       y = X$x[,2][which((Y == 0) & (FloodMag == 'N') & (years < 1962))], 
       col = 'black', pch = 1, ylim = c(-3,3), xlim = c(-5,5), ann = FALSE, axes = FALSE)
  legend('bottomleft', legend = c('Large', 'Moderate', 'Small', 'Unknown', 'No Flood or Not Large'),
         title = expression(bold('Flood Magnitude')), title.adj = 0.1,
         col = c('orange', 'green', 'blue', 'hotpink', 'black'), 
         pch = c(16,16,16,16,16), cex = .7, bty = 'n')
  legend('bottomright', legend = c('1962-2020', 'pre-1962'),
         title = expression(bold('Time Period')), title.adj = 0.1,
         col = gray(0.6), 
         pch = c(16,1), cex = .7, bty = 'n')
  par(xpd = TRUE)
  #text(x = -6.9, y = 3.2, 'More\nSnow')
  #text(x = 5, y = -4, 'Colder')
  par(xpd = FALSE)
  if(!is.null(label_mag)){
    if(label_mag == 'L'){
      text(x = X$x[,1][which((Y == 1) & (FloodMag == 'L') & (years < 1962))],
           y = X$x[,2][which((Y == 1) & (FloodMag == 'L') & (years < 1962))],
           paste0('L', seq(1,6,1)))
    }
    if(label_mag == 'M'){
      text(x = X$x[,1][which((Y == 1) & (FloodMag == 'M') & (years < 1962))],
           y = X$x[,2][which((Y == 1) & (FloodMag == 'M') & (years < 1962))],
           paste0('M', seq(1,5,1)))
    }
    if(label_mag == 'S'){
      text(x = X$x[,1][which((Y == 1) & (FloodMag == 'S') & (years < 1962))],
           y = X$x[,2][which((Y == 1) & (FloodMag == 'S') & (years < 1962))],
           paste0('S', seq(1,5,1)))
    }
    if(label_mag == 'U'){
      text(x = X$x[,1][which((FloodMag == 'U'))],
           y = X$x[,2][which((FloodMag == 'U'))],
           paste0('U', seq(1,4,1)))
    }
    if(label_mag == 'N'){
      text(x = X$x[,1][which((FloodMag == 'N') & (years < 1962))],
           y = X$x[,2][which((FloodMag == 'N') & (years < 1962))],
           paste0('N', seq(1,length(which((FloodMag == 'N') & (years < 1962))),1)), col = 'purple', cex = 0.7)
    }
  }
  #plot biplot axes for each variable
  arrows(x0 = 0, x1 = load_x, y0 = 0, y1 = load_y, col = "black", length = 0.1, lwd = 1.5)
  text(load_x, load_y, labels = c('','','DDF','Precipitation'), col  ="black", pos = text_pos, cex = 0.7)
  dev.off()
}


plot_EDA_PCA_MSFloodsPre1962_fill_animation = function(fname, X, X_label, Y_label, Y, FloodMag, years)
  {
  png(paste0(fname,'_1.png'), res = 300, units = 'in', width = 5, height = 5)
  par(mar = c(4,4,1,1))
  plot(x = X$x[,1], y = X$x[,2], pch = 16, ylim = c(-3,3), xlim = c(-5,5),
       ylab = Y_label, 
       xlab = X_label, 
       cex.lab = 0.9, 
       col = 'white', 
       axes = FALSE)
  axis(side = 1, at = seq(-5,5,1), labels = TRUE, cex.axis = 0.5)
  axis(side = 2, at = seq(-3,3,1), labels = TRUE, cex.axis = 0.5)
  box()
  par(new = TRUE)
  plot(x = X$x[,1][which((FloodMag == 'N' | FloodMag == 'M' | FloodMag == 'S') & (years >= 1962))], 
       y = X$x[,2][which((FloodMag == 'N' | FloodMag == 'M' | FloodMag == 'S') & (years >= 1962))], 
       col = 'black', pch = 16, ylim = c(-3,3), xlim = c(-5,5), ann = FALSE, axes = FALSE)
  par(new = TRUE)
  plot(x = X$x[,1][which((Y == 1) & (FloodMag == 'L') & (years >= 1962))], 
       y = X$x[,2][which((Y == 1) & (FloodMag == 'L') & (years >= 1962))], 
       col = 'orange', pch = 16, ylim = c(-3,3), xlim = c(-5,5), ann = FALSE, axes = FALSE)
  legend('bottomleft', legend = c('Large', 'Moderate', 'Small', 'Unknown', 'No Flood or Not Large'),
         title = expression(bold('Flood Magnitude')), title.adj = 0.1,
         col = c('orange', 'green', 'blue', 'hotpink', 'black'), 
         pch = c(16,16,16,16,16), cex = .7, bty = 'n')
  legend('bottomright', legend = c('1962-2020', 'pre-1962'),
         title = expression(bold('Time Period')), title.adj = 0.1,
         col = gray(0.6), 
         pch = c(16,1), cex = .7, bty = 'n')
  dev.off()
  
  png(paste0(fname,'_2.png'), res = 300, units = 'in', width = 5, height = 5)
  par(mar = c(4,4,1,1))
  plot(x = X$x[,1], y = X$x[,2], pch = 16, ylim = c(-3,3), xlim = c(-5,5),
       ylab = Y_label, 
       xlab = X_label, 
       cex.lab = 0.9, 
       col = 'white', 
       axes = FALSE)
  axis(side = 1, at = seq(-5,5,1), labels = TRUE, cex.axis = 0.5)
  axis(side = 2, at = seq(-3,3,1), labels = TRUE, cex.axis = 0.5)
  box()
  par(new = TRUE)
  plot(x = X$x[,1][which((FloodMag == 'N' | FloodMag == 'M' | FloodMag == 'S') & (years >= 1962))], 
       y = X$x[,2][which((FloodMag == 'N' | FloodMag == 'M' | FloodMag == 'S') & (years >= 1962))], 
       col = 'black', pch = 16, ylim = c(-3,3), xlim = c(-5,5), ann = FALSE, axes = FALSE)
  par(new = TRUE)
  plot(x = X$x[,1][which((Y == 1) & (FloodMag == 'L') & (years >= 1962))], 
       y = X$x[,2][which((Y == 1) & (FloodMag == 'L') & (years >= 1962))], 
       col = 'orange', pch = 16, ylim = c(-3,3), xlim = c(-5,5), ann = FALSE, axes = FALSE)
  par(new = TRUE)
  plot(x = X$x[,1][which((Y == 1) & (FloodMag == 'L') & (years < 1962))], 
       y = X$x[,2][which((Y == 1) & (FloodMag == 'L') & (years < 1962))], 
       col = 'orange', pch = 1, ylim = c(-3,3), xlim = c(-5,5), ann = FALSE, axes = FALSE)
  legend('bottomleft', legend = c('Large', 'Moderate', 'Small', 'Unknown', 'No Flood or Not Large'),
         title = expression(bold('Flood Magnitude')), title.adj = 0.1,
         col = c('orange', 'green', 'blue', 'hotpink', 'black'), 
         pch = c(16,16,16,16,16), cex = .7, bty = 'n')
  legend('bottomright', legend = c('1962-2020', 'pre-1962'),
         title = expression(bold('Time Period')), title.adj = 0.1,
         col = gray(0.6), 
         pch = c(16,1), cex = .7, bty = 'n')
  dev.off()
  
  png(paste0(fname,'_3.png'), res = 300, units = 'in', width = 5, height = 5)
  par(mar = c(4,4,1,1))
  plot(x = X$x[,1], y = X$x[,2], pch = 16, ylim = c(-3,3), xlim = c(-5,5),
       ylab = Y_label, 
       xlab = X_label, 
       cex.lab = 0.9, 
       col = 'white', 
       axes = FALSE)
  axis(side = 1, at = seq(-5,5,1), labels = TRUE, cex.axis = 0.5)
  axis(side = 2, at = seq(-3,3,1), labels = TRUE, cex.axis = 0.5)
  box()
  par(new = TRUE)
  plot(x = X$x[,1][which((FloodMag == 'N' | FloodMag == 'M' | FloodMag == 'S') & (years >= 1962))], 
       y = X$x[,2][which((FloodMag == 'N' | FloodMag == 'M' | FloodMag == 'S') & (years >= 1962))], 
       col = 'black', pch = 16, ylim = c(-3,3), xlim = c(-5,5), ann = FALSE, axes = FALSE)
  par(new = TRUE)
  plot(x = X$x[,1][which((Y == 1) & (FloodMag == 'L') & (years >= 1962))], 
       y = X$x[,2][which((Y == 1) & (FloodMag == 'L') & (years >= 1962))], 
       col = 'orange', pch = 16, ylim = c(-3,3), xlim = c(-5,5), ann = FALSE, axes = FALSE)
  par(new = TRUE)
  plot(x = X$x[,1][which((Y == 1) & (FloodMag == 'L') & (years < 1962))], 
       y = X$x[,2][which((Y == 1) & (FloodMag == 'L') & (years < 1962))], 
       col = 'orange', pch = 1, ylim = c(-3,3), xlim = c(-5,5), ann = FALSE, axes = FALSE)
  par(new = TRUE)
  plot(x = X$x[,1][which((Y == 1) & (FloodMag == 'M') & (years < 1962))], 
       y = X$x[,2][which((Y == 1) & (FloodMag == 'M') & (years < 1962))], 
       col = 'green', pch = 1, ylim = c(-3,3), xlim = c(-5,5), ann = FALSE, axes = FALSE)
  legend('bottomleft', legend = c('Large', 'Moderate', 'Small', 'Unknown', 'No Flood or Not Large'),
         title = expression(bold('Flood Magnitude')), title.adj = 0.1,
         col = c('orange', 'green', 'blue', 'hotpink', 'black'), 
         pch = c(16,16,16,16,16), cex = .7, bty = 'n')
  legend('bottomright', legend = c('1962-2020', 'pre-1962'),
         title = expression(bold('Time Period')), title.adj = 0.1,
         col = gray(0.6), 
         pch = c(16,1), cex = .7, bty = 'n')
  dev.off()
  
  png(paste0(fname,'_4.png'), res = 300, units = 'in', width = 5, height = 5)
  par(mar = c(4,4,1,1))
  plot(x = X$x[,1], y = X$x[,2], pch = 16, ylim = c(-3,3), xlim = c(-5,5),
       ylab = Y_label, 
       xlab = X_label, 
       cex.lab = 0.9, 
       col = 'white', 
       axes = FALSE)
  axis(side = 1, at = seq(-5,5,1), labels = TRUE, cex.axis = 0.5)
  axis(side = 2, at = seq(-3,3,1), labels = TRUE, cex.axis = 0.5)
  box()
  par(new = TRUE)
  plot(x = X$x[,1][which((FloodMag == 'N' | FloodMag == 'M' | FloodMag == 'S') & (years >= 1962))], 
       y = X$x[,2][which((FloodMag == 'N' | FloodMag == 'M' | FloodMag == 'S') & (years >= 1962))], 
       col = 'black', pch = 16, ylim = c(-3,3), xlim = c(-5,5), ann = FALSE, axes = FALSE)
  par(new = TRUE)
  plot(x = X$x[,1][which((Y == 1) & (FloodMag == 'L') & (years >= 1962))], 
       y = X$x[,2][which((Y == 1) & (FloodMag == 'L') & (years >= 1962))], 
       col = 'orange', pch = 16, ylim = c(-3,3), xlim = c(-5,5), ann = FALSE, axes = FALSE)
  par(new = TRUE)
  plot(x = X$x[,1][which((Y == 1) & (FloodMag == 'L') & (years < 1962))], 
       y = X$x[,2][which((Y == 1) & (FloodMag == 'L') & (years < 1962))], 
       col = 'orange', pch = 1, ylim = c(-3,3), xlim = c(-5,5), ann = FALSE, axes = FALSE)
  par(new = TRUE)
  plot(x = X$x[,1][which((Y == 1) & (FloodMag == 'M') & (years < 1962))], 
       y = X$x[,2][which((Y == 1) & (FloodMag == 'M') & (years < 1962))], 
       col = 'green', pch = 1, ylim = c(-3,3), xlim = c(-5,5), ann = FALSE, axes = FALSE)
  par(new = TRUE)
  plot(x = X$x[,1][which((Y == 1) & (FloodMag == 'S') & (years < 1962))], 
       y = X$x[,2][which((Y == 1) & (FloodMag == 'S') & (years < 1962))], 
       col = 'blue', pch = 1, ylim = c(-3,3), xlim = c(-5,5), ann = FALSE, axes = FALSE)
  legend('bottomleft', legend = c('Large', 'Moderate', 'Small', 'Unknown', 'No Flood or Not Large'),
         title = expression(bold('Flood Magnitude')), title.adj = 0.1,
         col = c('orange', 'green', 'blue', 'hotpink', 'black'), 
         pch = c(16,16,16,16,16), cex = .7, bty = 'n')
  legend('bottomright', legend = c('1962-2020', 'pre-1962'),
         title = expression(bold('Time Period')), title.adj = 0.1,
         col = gray(0.6), 
         pch = c(16,1), cex = .7, bty = 'n')
  dev.off()
  
  png(paste0(fname,'_5.png'), res = 300, units = 'in', width = 5, height = 5)
  par(mar = c(4,4,1,1))
  plot(x = X$x[,1], y = X$x[,2], pch = 16, ylim = c(-3,3), xlim = c(-5,5),
       ylab = Y_label, 
       xlab = X_label, 
       cex.lab = 0.9, 
       col = 'white', 
       axes = FALSE)
  axis(side = 1, at = seq(-5,5,1), labels = TRUE, cex.axis = 0.5)
  axis(side = 2, at = seq(-3,3,1), labels = TRUE, cex.axis = 0.5)
  box()
  par(new = TRUE)
  plot(x = X$x[,1][which((FloodMag == 'N' | FloodMag == 'M' | FloodMag == 'S') & (years >= 1962))], 
       y = X$x[,2][which((FloodMag == 'N' | FloodMag == 'M' | FloodMag == 'S') & (years >= 1962))], 
       col = 'black', pch = 16, ylim = c(-3,3), xlim = c(-5,5), ann = FALSE, axes = FALSE)
  par(new = TRUE)
  plot(x = X$x[,1][which((Y == 1) & (FloodMag == 'L') & (years >= 1962))], 
       y = X$x[,2][which((Y == 1) & (FloodMag == 'L') & (years >= 1962))], 
       col = 'orange', pch = 16, ylim = c(-3,3), xlim = c(-5,5), ann = FALSE, axes = FALSE)
  par(new = TRUE)
  plot(x = X$x[,1][which((Y == 1) & (FloodMag == 'L') & (years < 1962))], 
       y = X$x[,2][which((Y == 1) & (FloodMag == 'L') & (years < 1962))], 
       col = 'orange', pch = 1, ylim = c(-3,3), xlim = c(-5,5), ann = FALSE, axes = FALSE)
  par(new = TRUE)
  plot(x = X$x[,1][which((Y == 1) & (FloodMag == 'M') & (years < 1962))], 
       y = X$x[,2][which((Y == 1) & (FloodMag == 'M') & (years < 1962))], 
       col = 'green', pch = 1, ylim = c(-3,3), xlim = c(-5,5), ann = FALSE, axes = FALSE)
  par(new = TRUE)
  plot(x = X$x[,1][which((Y == 1) & (FloodMag == 'S') & (years < 1962))], 
       y = X$x[,2][which((Y == 1) & (FloodMag == 'S') & (years < 1962))], 
       col = 'blue', pch = 1, ylim = c(-3,3), xlim = c(-5,5), ann = FALSE, axes = FALSE)
  par(new = TRUE)
  plot(x = X$x[,1][which((FloodMag == 'U'))], 
       y = X$x[,2][which((FloodMag == 'U'))], 
       col = 'hotpink', pch = 1, ylim = c(-3,3), xlim = c(-5,5), ann = FALSE, axes = FALSE)
  legend('bottomleft', legend = c('Large', 'Moderate', 'Small', 'Unknown', 'No Flood or Not Large'),
         title = expression(bold('Flood Magnitude')), title.adj = 0.1,
         col = c('orange', 'green', 'blue', 'hotpink', 'black'), 
         pch = c(16,16,16,16,16), cex = .7, bty = 'n')
  legend('bottomright', legend = c('1962-2020', 'pre-1962'),
         title = expression(bold('Time Period')), title.adj = 0.1,
         col = gray(0.6), 
         pch = c(16,1), cex = .7, bty = 'n')
  dev.off()
  
  png(paste0(fname,'_6.png'), res = 300, units = 'in', width = 5, height = 5)
  par(mar = c(4,4,1,1))
  plot(x = X$x[,1], y = X$x[,2], pch = 16, ylim = c(-3,3), xlim = c(-5,5),
       ylab = Y_label, 
       xlab = X_label, 
       cex.lab = 0.9, 
       col = 'white', 
       axes = FALSE)
  axis(side = 1, at = seq(-5,5,1), labels = TRUE, cex.axis = 0.5)
  axis(side = 2, at = seq(-3,3,1), labels = TRUE, cex.axis = 0.5)
  box()
  par(new = TRUE)
  plot(x = X$x[,1][which((FloodMag == 'N' | FloodMag == 'M' | FloodMag == 'S') & (years >= 1962))], 
       y = X$x[,2][which((FloodMag == 'N' | FloodMag == 'M' | FloodMag == 'S') & (years >= 1962))], 
       col = 'black', pch = 16, ylim = c(-3,3), xlim = c(-5,5), ann = FALSE, axes = FALSE)
  par(new = TRUE)
  plot(x = X$x[,1][which((Y == 1) & (FloodMag == 'L') & (years >= 1962))], 
       y = X$x[,2][which((Y == 1) & (FloodMag == 'L') & (years >= 1962))], 
       col = 'orange', pch = 16, ylim = c(-3,3), xlim = c(-5,5), ann = FALSE, axes = FALSE)
  par(new = TRUE)
  plot(x = X$x[,1][which((Y == 1) & (FloodMag == 'L') & (years < 1962))], 
       y = X$x[,2][which((Y == 1) & (FloodMag == 'L') & (years < 1962))], 
       col = 'orange', pch = 1, ylim = c(-3,3), xlim = c(-5,5), ann = FALSE, axes = FALSE)
  par(new = TRUE)
  plot(x = X$x[,1][which((Y == 1) & (FloodMag == 'M') & (years < 1962))], 
       y = X$x[,2][which((Y == 1) & (FloodMag == 'M') & (years < 1962))], 
       col = 'green', pch = 1, ylim = c(-3,3), xlim = c(-5,5), ann = FALSE, axes = FALSE)
  par(new = TRUE)
  plot(x = X$x[,1][which((Y == 1) & (FloodMag == 'S') & (years < 1962))], 
       y = X$x[,2][which((Y == 1) & (FloodMag == 'S') & (years < 1962))], 
       col = 'blue', pch = 1, ylim = c(-3,3), xlim = c(-5,5), ann = FALSE, axes = FALSE)
  par(new = TRUE)
  plot(x = X$x[,1][which((FloodMag == 'U'))], 
       y = X$x[,2][which((FloodMag == 'U'))], 
       col = 'hotpink', pch = 1, ylim = c(-3,3), xlim = c(-5,5), ann = FALSE, axes = FALSE)
  par(new = TRUE)
  plot(x = X$x[,1][which((Y == 0) & (FloodMag == 'N') & (years < 1962))], 
       y = X$x[,2][which((Y == 0) & (FloodMag == 'N') & (years < 1962))], 
       col = 'black', pch = 1, ylim = c(-3,3), xlim = c(-5,5), ann = FALSE, axes = FALSE)
  legend('bottomleft', legend = c('Large', 'Moderate', 'Small', 'Unknown', 'No Flood or Not Large'),
         title = expression(bold('Flood Magnitude')), title.adj = 0.1,
         col = c('orange', 'green', 'blue', 'hotpink', 'black'), 
         pch = c(16,16,16,16,16), cex = .7, bty = 'n')
  legend('bottomright', legend = c('1962-2020', 'pre-1962'),
         title = expression(bold('Time Period')), title.adj = 0.1,
         col = gray(0.6), 
         pch = c(16,1), cex = .7, bty = 'n')
  dev.off()
}
