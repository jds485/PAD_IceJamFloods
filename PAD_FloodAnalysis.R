#Statistical Analysis for the Comment on Beltaos (2018)
#Code Authors: Jared D. Smith (jds485@cornell.edu)
#            : Jonathan Lamontagne (Jonathan.Lamontagne@tufts.edu)

#Blurb on sections in the code----

#Set working directory----
setwd("C:\\Users\\jsmif\\Documents\\Cornell\\Research\\Publications\\CountRegressionIceJams\\DataAnalysis\\PAD_IceJamFloods")

#Load libraries----
library(Hmisc)

#Load data----
#All flood data
Floods = read.csv(file = 'FloodData.csv', stringsAsFactors = FALSE)
#Select only moderate (M) and extreme (X) floods (scale values of 2 and 3), and Beltaos floods
#NOTE: JDS added in 0s for missing catalog data based on sediment cores in Wolfe (2006). Years 1927 and 1928 were given 1s because of flooding at PAD54.
MX = Floods[,c('YEAR', 'Floods23only', 'Beltaos')]

#Compute cumulative number of floods for replication of Beltaos
MX$FldSum = 0
MX$BeltSum = 0
for (i in 1:nrow(MX)){
  MX$FldSum[i] = sum(MX$Floods23only[seq(1,i,1)], na.rm = TRUE)
  MX$BeltSum[i] = sum(MX$Beltaos[seq(1,i,1)], na.rm = TRUE)
}
rm(i)
#Because Beltaos started in 1900 at a cumulative flood count of 2, need to adjust the cumulative number by the number observed in 1900, plus 2
MX$FldSum = MX$FldSum - MX$FldSum[MX$YEAR == 1900] + 2
#Also need to add one to Beltaos cumulative (must have considered a flood before 1900 but did not show where it is)
MX$BeltSum[MX$YEAR >= 1900] = MX$BeltSum[MX$YEAR >= 1900] + 1

#Plot of flood data----
png('CumulativeFloods.png', res = 300, units = 'in', width = 6, height = 6)
par(mar = c(5,5,3,1), xaxs = 'i', yaxs = 'i')
#Plot filler data: Both no floods
plot(MX$YEAR[MX$Beltaos == 0 & MX$Floods23only == 0], MX$FldSum[MX$Beltaos == 0 & MX$Floods23only == 0], pch = 1,
     ylim = c(-6,28), xlim = c(MX$YEAR[1], max(MX$YEAR)),
     xlab = '', ylab = '', axes = FALSE)

#Plot Floods not considered by Beltaos
par(new = TRUE)
plot(MX$YEAR[MX$Beltaos == 0 & MX$Floods23only == 1], MX$FldSum[MX$Beltaos == 0 & MX$Floods23only == 1], col = 'red', pch = 16,
     ylim = c(-6,28), xlim = c(MX$YEAR[1], max(MX$YEAR)),
     xlab = 'Year', ylab = 'Cumulative Number of Floods Since 1888',
     cex.axis = 1.5, cex.lab = 1.5)
#Plot all other floods considered in both studies
par(new = TRUE)
plot(MX$YEAR[MX$Beltaos == 1 & MX$Floods23only == 1], MX$FldSum[MX$Beltaos == 1 & MX$Floods23only == 1], col = 'blue', pch = 16,
     ylim = c(-6,28), xlim = c(MX$YEAR[1], max(MX$YEAR)),
     xlab = '', ylab = '', axes = FALSE)


#Polygon for dam filling years
polygon(x = c(1968, 1968, 1971, 1971), y = c(-10, 28, 28, -10), col = 'grey', density = 0, lwd = 2)

minor.tick(nx = 5, ny = 5, tick.ratio = 0.5)
legend('topleft', legend = c('Flood Not in Beltaos (2018)', 'Flood in Beltaos (2018)', 'No Flood', 'Reservoir Filling'), pch = c(16,16,1,NA), lty = c(NA,NA,NA,1), lwd = 1, col = c('red', 'blue', 'black', 'grey'), cex = 1.2)

dev.off()

#Pacific Decadal Osciallation indices
#Monthly index for the PDO
PDOi = read.csv('PDO.csv', stringsAsFactors = FALSE)
#Annual sea surface temperature and sea level pressure
PDO_SST_SLP = read.csv('PDO_SLP_SST.csv', stringsAsFactors = FALSE)

#FIXME: Process monthly data to a usable format

#Binomial exact tests----
# Compare to Beltaos' analysis blocks for 1900 - 2017 data----
#Estimate binomial p for the base case of pre-dam in 1968
# Years 1826 - 1967 - All historical pre-dam construction
p1826 = sum(MX[MX$YEAR <= 1967, 'Floods23only'], na.rm = TRUE)/length(which(is.na(MX[MX$YEAR <= 1967, 'Floods23only']) == FALSE))
# Years 1900 - 1967
p1900 = sum(MX[MX$YEAR >= 1900 & MX$YEAR <= 1967, 'Floods23only'], na.rm = TRUE)/length(which(is.na(MX[MX$YEAR >= 1900 & MX$YEAR <= 1967, 'Floods23only']) == FALSE))
# Years 1910 - 1967
p1910 = sum(MX[MX$YEAR >= 1910 & MX$YEAR <= 1967, 'Floods23only'], na.rm = TRUE)/length(which(is.na(MX[MX$YEAR >= 1910 & MX$YEAR <= 1967, 'Floods23only']) == FALSE))
# Years 1922 - 1967
p1922 = sum(MX[MX$YEAR >= 1922 & MX$YEAR <= 1967, 'Floods23only'], na.rm = TRUE)/length(which(is.na(MX[MX$YEAR >= 1922 & MX$YEAR <= 1967, 'Floods23only']) == FALSE))
# Years 1930 - 1967
p1930 = sum(MX[MX$YEAR >= 1930 & MX$YEAR <= 1967, 'Floods23only'], na.rm = TRUE)/length(which(is.na(MX[MX$YEAR >= 1930 & MX$YEAR <= 1967, 'Floods23only']) == FALSE))
# Years 1940 - 1967
p1940 = sum(MX[MX$YEAR >= 1940 & MX$YEAR <= 1967, 'Floods23only'], na.rm = TRUE)/length(which(is.na(MX[MX$YEAR >= 1940 & MX$YEAR <= 1967, 'Floods23only']) == FALSE))
# Years 1950 - 1967
p1950 = sum(MX[MX$YEAR >= 1950 & MX$YEAR <= 1967, 'Floods23only'], na.rm = TRUE)/length(which(is.na(MX[MX$YEAR >= 1950 & MX$YEAR <= 1967, 'Floods23only']) == FALSE))

#Binomial exact tests using those estimates of p as the null hypothesis value.
#Compute number of floods post-dam (FldPD) starting in 1971, and the length of record from 1971 - 2017
FldPD = sum(MX[MX$YEAR >= 1971 & MX$YEAR <= 2017, 'Floods23only'], na.rm = TRUE)
n = length(which(is.na(MX[MX$YEAR >= 1971 & MX$YEAR <= 2017, 'Floods23only']) == FALSE))

#Probability of seeing FldPD floods or less, given p from above. This is a test on p being less than the p above, given n years with FldPD floods.
# Years 1826 - 1967
bt1826 = binom.test(x = FldPD, n = n, p = p1826, alternative = 'less')
# Years 1900 - 1967
bt1900 = binom.test(x = FldPD, n = n, p = p1900, alternative = 'less')
# Years 1910 - 1967
bt1910 = binom.test(x = FldPD, n = n, p = p1910, alternative = 'less')
# Years 1922 - 1967
bt1922 = binom.test(x = FldPD, n = n, p = p1922, alternative = 'less')
# Years 1930 - 1967
bt1930 = binom.test(x = FldPD, n = n, p = p1930, alternative = 'less')
# Years 1940 - 1967
bt1940 = binom.test(x = FldPD, n = n, p = p1940, alternative = 'less')
# Years 1950 - 1967
bt1950 = binom.test(x = FldPD, n = n, p = p1950, alternative = 'less')

# Moving window analysis using all of the historical data----
#Window size in years
size = 50
#Window jump size in years
jump = 1
#FIXME: Evaluate the sensitivity of this analysis wrt size. Should pick the size that stabilizes the values.

#FIXME: Currently does not drop reservior filling years.

#Compute p-values of binomial exact tests. Treat the first 'size' years of data as the base case.
#Preallocate p value vector and midpoint vector. First window is base case and does not get tested.
pvec = mids = vector('numeric', length = (nrow(MX) - size))
for (i in 1:(length(pvec) + 1)){
  #Number of floods in size
  x = sum(MX[MX$YEAR >= (MX$YEAR[1] + (i-1)) & MX$YEAR <= (MX$YEAR[1] + (i-1) + size - 1), 'Floods23only'], na.rm = TRUE)
  #Number of years with data in size
  nx = length(which(is.na(MX[MX$YEAR >= (MX$YEAR[1] + (i-1)) & MX$YEAR <= (MX$YEAR[1] + (i-1) + size - 1), 'Floods23only']) == FALSE))
  
  #Extract the p-value from the lower tail binomial test.
  if (i == 1){
    #Base case. Estimate p from the data, and use as the null value in tests that follow.
    pbase = x/nx
  }
  else{
    pvec[i-1] = binom.test(x = x, n = nx, p = pbase, alternative = 'less')[[3]]
    #Compute the midpoint of the window for plotting
    mids[i-1] = sum(MX[MX$YEAR >= (MX$YEAR[1] + (i-1)) & MX$YEAR <= (MX$YEAR[1] + (i-1) + size - 1), 'YEAR'][c(1,size)])/2
  }
}
rm(x, nx)

#Plot of moving window
png('MovingWindowBinomialTest.png', res = 300, width = 8, height = 8, units = 'in')
par(mar = c(4,4,4,1), xaxs = 'i', yaxs = 'i')
#Plot of the p-values. X location is the midpoint of the window
plot(x = mids, y = pvec, pch = 16, lty = 1, type = 'o',
     xlim = c(MX$YEAR[1], MX$YEAR[nrow(MX)]), ylim = c(0, 1),
     xlab = 'Year', ylab = 'p-value', main = 'Binomial Test p-values for a 50-Year Moving Window \n Base Years: 1826 - 1875',
     cex.axis = 1.5, cex.main = 1.5, cex.lab = 1.5)

axis(side = 1, at = seq(1830,2010,10), labels = FALSE)
axis(side = 2, at = seq(0.1,0.9,0.2), labels = FALSE)

#Line for 0.05 significance
lines(x = c(MX$YEAR[1], MX$YEAR[nrow(MX)]), y = c(0.05, 0.05), lty = 2, lwd = 2)

#Polygon for dam filling years
polygon(x = c(1968, 1968, 1971, 1971), y = c(0, 1, 1, 0), col = 'grey', density = 0, lwd = 2)

par(new = TRUE)
#Plot of the floods to show where they exist
plot(x = MX$YEAR[MX$Floods23only == 1], y = rep(0.01,length(MX$YEAR[MX$Floods23only == 1])), 
     pch = 1, col = 'purple',
     xlim = c(MX$YEAR[1], MX$YEAR[nrow(MX)]), ylim = c(0, 1),
     xlab = '', ylab = '', main = '', axes = FALSE)

#par(new = TRUE)
#plot of the no data years to show where no data exist - Filled in with stratigraphic column
#plot(x = MX$YEAR[is.na(MX$Floods23only)], y = rep(0.01,length(MX$YEAR[is.na(MX$Floods23only)])), 
#     pch = 'x', col = 'grey',
#     xlim = c(MX$YEAR[1], MX$YEAR[nrow(MX)]), ylim = c(0, 1),
#     xlab = '', ylab = '', main = '', axes = FALSE)

#Polygon for base years
polygon(x = c(MX$YEAR[1], MX$YEAR[1], MX$YEAR[size], MX$YEAR[size]), y = c(0, 0.1, 0.1, 0), col = 'red', density = 0, lwd = 2)

legend('topleft', legend = c('p-values', 'Floods', 'Dam filling', 'Base Years', '0.05 sig. level'), pch = c(16,1,22,22,NA), lty = c(1,NA,NA,NA,2), lwd = 2, col = c('black', 'purple', 'grey', 'red', 'black'), cex = 1.2)

dev.off()

#Geometric probability of seeing no ice jam floods in 20 or more years----
#Rs geometric distribution starts at x = 0, rather than the traditional x = 1. So, 20 - 2 is needed here.
Geom20 = 1 - pgeom(q = 20 - 2, prob = pbase)

#Moving window for geometric
#Using the same base period for p as for binomial p, what is the probability of waiting as many years as elapsed before the next flood?
# Select only those years with floods after the base period.
MX_Geom = MX[which(MX$Floods23only == 1 & MX$YEAR > (MX$YEAR[1] + size)),]
GeomProb = vector('numeric', length = nrow(MX_Geom)-1)
for (i in 1:length(GeomProb)){
  #Number of years before next flood
  x = MX_Geom$YEAR[i + 1] - MX_Geom$YEAR[i]
  
  #Compute the upper tail from geometric distribution for waiting at least x years for next flood given pbase probability in any year
  GeomProb[i] = 1 - pgeom(q = (x-2), prob = pbase)
}
rm(x,i)

png('MovingWindowGeometricWaitingTime.png', res = 300, width = 8, height = 8, units = 'in')
par(mar = c(4,6,4,1), xaxs = 'i', yaxs = 'i')
#Plot of the probabilities. X location is the flood year counted as 0 in geometric distribution waiting time
plot(x = MX_Geom$YEAR[-nrow(MX_Geom)], y = GeomProb, pch = 16,
     xlim = c(MX$YEAR[1], MX$YEAR[nrow(MX)]), ylim = c(0, 1),
     xlab = 'Year', ylab = 'Probability of Waiting \n at Least as Long as Observed', main = 'Geometric Waiting Time for Next Observed Flood \n Base Years: 1826 - 1875',
     cex.axis = 1.5, cex.main = 1.5, cex.lab = 1.5)

axis(side = 1, at = seq(1830,2010,10), labels = FALSE)
axis(side = 2, at = seq(0.1,0.9,0.2), labels = FALSE)

#Line for 0.05 significance
lines(x = c(MX$YEAR[1], MX$YEAR[nrow(MX)]), y = c(0.05, 0.05), lty = 2, lwd = 2)

#Polygon for dam filling years
polygon(x = c(1968, 1968, 1971, 1971), y = c(0, 1, 1, 0), col = 'grey', density = 0, lwd = 2)

par(new = TRUE)
#Plot of the floods to show where they exist
plot(x = MX$YEAR[MX$Floods23only == 1], y = rep(0.01,length(MX$YEAR[MX$Floods23only == 1])), 
     pch = 1, col = 'purple',
     xlim = c(MX$YEAR[1], MX$YEAR[nrow(MX)]), ylim = c(0, 1),
     xlab = '', ylab = '', main = '', axes = FALSE)

#par(new = TRUE)
#plot of the no data years to show where no data exist
#plot(x = MX$YEAR[is.na(MX$Floods23only)], y = rep(0.01,length(MX$YEAR[is.na(MX$Floods23only)])), 
#     pch = 'x', col = 'grey',
#     xlim = c(MX$YEAR[1], MX$YEAR[nrow(MX)]), ylim = c(0, 1),
#     xlab = '', ylab = '', main = '', axes = FALSE)

#Polygon for base years
polygon(x = c(MX$YEAR[1], MX$YEAR[1], MX$YEAR[size], MX$YEAR[size]), y = c(0, 0.1, 0.1, 0), col = 'red', density = 0, lwd = 2)

legend('topleft', legend = c('Probabilities', 'Floods', 'Dam filling', 'Base Years', '0.05 sig. level'), pch = c(16,1,22,22,NA), lty = c(NA,NA,NA,NA,2), lwd = 2, col = c('black', 'purple', 'grey','red', 'black'), cex = 1.2)

dev.off()

#Replication of Beltaos' regression analysis tests----
# Years 1900 - 1967
lm1900 = lm(MX[MX$YEAR >= 1900 & MX$YEAR <= 1967, 'BeltSum'] ~ MX[MX$YEAR >= 1900 & MX$YEAR <= 1967, 'YEAR'])
# Years 1910 - 1967
lm1910 = lm(MX[MX$YEAR >= 1910 & MX$YEAR <= 1967, 'BeltSum'] ~ MX[MX$YEAR >= 1910 & MX$YEAR <= 1967, 'YEAR'])
# Years 1922 - 1967
lm1922 = lm(MX[MX$YEAR >= 1922 & MX$YEAR <= 1967, 'BeltSum'] ~ MX[MX$YEAR >= 1922 & MX$YEAR <= 1967, 'YEAR'])
# Years 1930 - 1967
lm1930 = lm(MX[MX$YEAR >= 1930 & MX$YEAR <= 1967, 'BeltSum'] ~ MX[MX$YEAR >= 1930 & MX$YEAR <= 1967, 'YEAR'])
# Years 1940 - 1967
lm1940 = lm(MX[MX$YEAR >= 1940 & MX$YEAR <= 1967, 'BeltSum'] ~ MX[MX$YEAR >= 1940 & MX$YEAR <= 1967, 'YEAR'])
# Years 1950 - 1967
lm1950 = lm(MX[MX$YEAR >= 1950 & MX$YEAR <= 1967, 'BeltSum'] ~ MX[MX$YEAR >= 1950 & MX$YEAR <= 1967, 'YEAR'])
# Post-dam years: 1971 - 2017
lmPD = lm(MX[MX$YEAR >= 1971, 'BeltSum'] ~ MX[MX$YEAR >= 1971, 'YEAR'])

#Regression diagnostic plots - concerns about heteroskedasticity and normality. Showing one example.
plot(lm1900)

#Regression diagnostics plots
png('BeltaosRegressionDiagnostics.png', res = 300, units = 'in', width = 8, height = 6)
layout(cbind(1,2))
par(mar = c(4,5,3,1), xaxs = 'i', yaxs = 'i')
#Normal QQ plot for residuals
qqnorm(lmPD$residuals, ylim = c(-1.5,1.5), xlim = c(-2.5,2.5), main = 'Regression for 1971 - 2018 \n Normal Q-Q Plot', cex.lab = 1.5, cex.axis = 1.5)
qqline(lmPD$residuals)

#Residuals vs. Fitted Y Values
plot(lmPD$fitted.values, lmPD$residuals, type = 'p', xlim = c(13,17), ylim = c(-1.5,1.5), cex.lab = 1.5, cex.axis = 1.5,
     xlab = 'Fitted Y Values', ylab = 'Regression Residuals', main = 'Regression for 1971 - 2018 \n Residuals vs. Fitted Y Values')
lines(c(10,20), c(0,0), lty = 2)
dev.off()

png('BeltosRegressionCheck.png', res = 300, units = 'in', width = 7, height = 7)
par(mar = c(4,5,3,1), xaxs = 'i', yaxs = 'i')
plot(MX$YEAR, MX$BeltSum, 
     ylim = c(-1,23), xlim = c(1900, max(MX$YEAR)),
     xlab = 'Year', ylab = 'Beltaos (2018) Cumulative Number of Floods',
     cex.axis = 1.5, cex.lab = 1.5)
#Polygon for dam filling years
polygon(x = c(1968, 1968, 1971, 1971), y = c(-1, 31, 31, -1), col = 'grey', density = 0, lwd = 2)

#Add regression lines
abline(lm1900, col = 'red')
abline(lm1910, col = 'orange')
abline(lm1922, col = 'green')
abline(lm1930, col = 'blue')
abline(lm1940, col = 'purple')
abline(lm1950, col = 'pink')
abline(lmPD, col = 'black')

dev.off()

#Evaluation of PDO and Floods----
plot(seq(1856, 1990,1), PDO_SST_SLP$PDO_SST, type = 'o', pch = 16, col = 'red',
     xlim = c(1856,1990), ylim = c(-0.5, 0.5),
     xlab = 'Year', ylab = 'Seal Level Pressure or Temperature Anomaly', main = 'PDO and Moderate or Extreme Floods in PAD',
     cex.lab = 1.5, cex.main = 2, cex.axis = 1.5)
par(new = TRUE)
plot(seq(1856, 1990,1), PDO_SST_SLP$PDO_SLP, type = 'o', pch = 16, col = 'blue',
     xlim = c(1856,1990), ylim = c(-0.5, 0.5),
     xlab = '', ylab = '', main = '', axes = FALSE)
par(new = TRUE)
plot(x = MX$YEAR[MX$Floods23only == 1], y = rep(0.5,length(MX$YEAR[MX$Floods23only == 1])), 
     pch = 1,
     xlim = c(1856,1990), ylim = c(-0.5, 0.5),
     xlab = '', ylab = '', main = '', axes = FALSE)

#AR Model----

#Block Bootstrapping----