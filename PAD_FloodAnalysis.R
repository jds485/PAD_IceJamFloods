#Statistical Analysis for the Comment on Beltaos (2018)
#Code Authors: Jared D. Smith (jds485@cornell.edu)
#            : Jonathan Lamontagne (Jonathan.Lamontagne@tufts.edu)

#Blurb on sections in the code----

#Set working directory----
setwd("C:\\Users\\jsmif\\Documents\\Cornell\\Research\\Publications\\CountRegressionIceJams\\DataAnalysis\\PAD_IceJamFloods")

#Load libraries----

#Load data----
Floods = read.csv(file = 'FloodData.csv', stringsAsFactors = FALSE)

#Select only moderate (M) and extreme (X) floods (scale values of 2 and 3)
MX = Floods[,c('YEAR', 'Floods23only')]

#FIXME: remove years without flood data? currently using na.rm and is.na to check for NA data (no flood record) years

#Binomial exact tests----
# Compare to Beltaos' analysis blocks for 1900 - 2017 data----
#Estimate binomial p for the base case of pre-dam in 1968
# Years 1826 - 1967
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
#Compute number of floods post-dam starting in 1971, and the length of record from 1971 - 2017
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
#FIXME: Evaluate the sensitivity of this analysis wrt size and jump. Should pick the size and jump that stabilizes the values.

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
polygon(x = c(1968, 1968, 1971, 1971), y = c(0, 1, 1, 0), col = 'blue', density = 0, lwd = 2)

par(new = TRUE)
#Plot of the floods to show where they exist
plot(x = MX$YEAR[MX$Floods23only == 1], y = rep(0.01,length(MX$YEAR[MX$Floods23only == 1])), 
     pch = 1, col = 'red',
     xlim = c(MX$YEAR[1], MX$YEAR[nrow(MX)]), ylim = c(0, 1),
     xlab = '', ylab = '', main = '', axes = FALSE)

par(new = TRUE)
#plot of the no data years to show where no data exist
plot(x = MX$YEAR[is.na(MX$Floods23only)], y = rep(0.01,length(MX$YEAR[is.na(MX$Floods23only)])), 
     pch = 'x', col = 'grey',
     xlim = c(MX$YEAR[1], MX$YEAR[nrow(MX)]), ylim = c(0, 1),
     xlab = '', ylab = '', main = '', axes = FALSE)

#Polygon for base years
polygon(x = c(MX$YEAR[1], MX$YEAR[1], MX$YEAR[size], MX$YEAR[size]), y = c(0, 0.1, 0.1, 0), col = 'red', density = 0, lwd = 2)

legend('topleft', legend = c('p-values', 'Floods', 'No Data', 'Dam filling', 'Base Years', '0.05 sig. level'), pch = c(16,1,4,22,22,NA), lty = c(1,NA,NA,NA,NA,2), lwd = 2, col = c('black', 'red', 'grey', 'blue','red', 'black'), cex = 1.2)

dev.off()

#Replication of Beltaos' regression analysis tests----

# Years 1900 - 1967

# Years 1910 - 1967

# Years 1922 - 1967

# Years 1930 - 1967

# Years 1940 - 1967

# Years 1950 - 1967
