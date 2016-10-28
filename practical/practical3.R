# 1.1
cloud =read.table("P:/R Data/cloud.seeding.txt", header = T)

# 1.2
summary(cloud)
str(cloud)

# 1.3
cor(cloud, cloud$Y)
# most of them are not related to the Y

# 1.4
plot(cloud)
# we get the same conclusion as previous steps, we didn't see a very clear trend 
# between Y and other varibales

#1.5
cloud$A = as.factor(cloud$A)
cloud$E = as.factor(cloud$E)

plot(cloud$A, cloud$Y)
plot(cloud$E, cloud$Y)

# factor E shows a different distribution between two different distribution for
# Y, but A doesn't affect too much

#2.1
mT = lm(Y~T, data = cloud)
mT$coefficients
mT$residuals
mT$fitted.values

#2.2
summary(mT)
# The significance of the coefficient.

#2.3
# the rainfall doesn't increasing with time, because from the lm, the coefficient
# is negative, which indecates rainflass are decreasing with T, from the summary
# we see that the t test for significant of coffecient is 0.1365, which is smaller#
# than 0.05, we can conclude that the cofficient is significant as 0.05 significant
# level

#2.4
mC = lm(Y~C, data = cloud)
mP = lm(Y~P, data = cloud)

summary(mC)
summary(mP)
# both are not significant

#2.5
library(MASS)

new_mC = lm(Y~(C^2), data = cloud)
summary(new_mC)
# no

summary(lm(Y~log(C), data = cloud))
#better
summary(lm(Y~(P^2), data = cloud))
#no
summary(lm(Y~log(P), data = cloud))
#better

#2.6
summary(lm(log(Y)~C, data = cloud))

#2.7
mCPT = lm(Y~T+C+P, data = cloud)
summary(mCPT)

#2.8
mACPT = lm(Y~A+T+C+P, data = cloud)
summary(mACPT)
# A = 1 which is include cofficient A1, A=0, which is not include cofficient A1

#2.9
mall = lm(Y~A*(E+T+C+P+SNe), data = cloud)
summary(mall)

#3.1
par(mfrow = c(2,2))
plot(mCPT)

#3.2
# not really

#3.3
#leverage and influence

#3.4
newcloud = cloud[-c(1,2,15),]
mCPT = lm(Y~T+C+P, data = newcloud)
summary(mCPT)
