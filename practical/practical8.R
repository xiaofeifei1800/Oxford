#########################################   1
rats = read.csv("P:/R Data/rats.csv", header = T)

# take a initial look of data, and check if it is loaded correctly
str(rats)
# will consider does as one exploration varible

##########################################  2
#turn to dead rate
rats$dead_rate = rats$numdead/25
  
attach(rats)

par(mar=c(5,5,1,1))
plot(rats[rats$sex=='M'&rats$standard==1,"dose"],rats[rats$sex=='M'&
    rats$standard==1,"dead_rate"],pch=1,col='blue',xlab='Dose level',
    ylab='mortality', ylim=c(0,1))

points(rats[rats$sex=='M'&rats$standard==0,"dose"],rats[rats$sex=='M'&
                        rats$standard==0,"dead_rate"],pch=0,col='blue')
points(rats[rats$sex=='F'&rats$standard==1,"dose"],rats[rats$sex=='F'&
                        rats$standard==1,"dead_rate"],pch=2,col='red')
points(rats[rats$sex=='F'&rats$standard==0,"dose"],rats[rats$sex=='F'&
                        rats$standard==0,"dead_rate"],pch=5,col='red')
legend("topright", col = c('blue', 'blue', 'red','red'), pch = c(1,0,2,5),
       legend = c("Male and Standard", "Male and Non Standard", 
                "Female and Standard", "Female and Non Standard"))

# The difference between standard zore and stadard one is not so much different,
# so we remove the standard from the plot, and look at mortality respect to does 
# and sex

plot(dose, dead_rate, col = standard+1, pch = 16, xlab='Dose level', 
     ylab='mortality', ylim=c(0,1))
#we see the the numdead decrease first, as dose increase, and increase after, 
#it may follows a quadratic, the sex and dose have affect, but there is not a 
#clear relation.

boxplot(dead_rate ~ standard + sex, ylab='mortality', ylim=c(0,1))

###########################################  3

library(MASS)

# we see a quadratic relationship between does and dead-rate, so apply quadratic 
#model
model1 = glm(cbind(numdead,numalive)~I(dose^2)+dose,family=binomial(),data=rats)
summary(model1)
1-pchisq(20.44,21)
#favor current model

###########################################  4
#Comment on your model and interpret the parameters giving suitable measures of 
#uncertainty

#stepAIC#
# put all other varibles, and do stepwise with AIC
model2 = glm(cbind(numdead,numalive)~I(dose^2)+dose+sex+standard,
             family=binomial(),data=rats)
stp = stepAIC(model2,
              scope = list(upper = ~sex * standard * dose * I(dose^2),
                           lower = ~1))
# StepAIC gives us model3
model3 = glm(cbind(numdead, numalive) ~ I(dose^2) + dose + sex + I(dose^2):sex + 
               dose:sex + I(dose^2):dose, family=binomial(),data=rats)
# look at the nested deviance tests, drop I(dose^2)*dose
anova(model3, test = "Chisq")

# we get model4
model4 = glm(cbind(numdead, numalive) ~ I(dose^2) + dose + sex + I(dose^2):sex + 
               dose:sex, family=binomial(),data=rats)

# I don't know if I should do it
#anova(justice.glm<-glm(formula = cbind(numdead,numalive)~I(dose^2) * sex * 
#standard * dose, family = binomial(), data = rats),test='Chisq')

# compare the current model with the model we find last step
anova(model1, model4, test = "Chisq")

exp(confint(model4))
###########################################

summary(model4)
1 - pchisq(6.8287, 18)
#model looks OK RD=6.8287 on 18 DOF

rstandard(model4)
sum(abs(rstandard(model4)) >= 2) #no big numbers - so OK (-2,2)

#Working Residuals
par(mar=c(5,5,1,1))
plot(model4$fitted.values,model4$residuals,xlab='Fitted Values', 
     ylab='Working Residuals', pch=19,xlim=c(0,1),ylim=c(-1.5,1.5))
abline(h=0)
#ok

# LEVERAGE
plot(influence(model4)$hat,ylim=c(0,1.5),col='blue',
     pch=19,ylab='Leverage')
abline(h = 2*(6/24))
# ok

# DEVIANCE RESIDUALS
plot(model4$fitted.values,rstandard(model4),xlab='Fitted Values',
     ylab='Deviance Residuals',pch=19,xlim=c(0,1),ylim=c(-2,2))
abline(h=0)
#ok

qqnorm(rstandard(model4),pch=19,main='')
qqline(rstandard(model4))
#ok

# COOKS DISTANCE
plot(cooks.distance(model4), ylim=c(0,0.7),
     pch=19,ylab="Cook's Distance")
abline(h = 8/(24-2*6), col = "red")
#ok
############################################
x <- seq(from=0,to=7,length.out=20)
ndM <- data.frame(dose=x,sex="M")
bw.respM <- predict(model4,ndM,type='response')
plot(ndM$dose,bw.respM*20,type='l',col='blue',ylim=c(0,20), xlab='Dose level', 
     ylab='Proportion dead')
points(rats[rats$sex=="M","dose"],rats[rats$sex=="M","numdead"],pch=19,
       col='blue')
ndF <- data.frame(dose=x,sex="F")
bw.respF <- predict(model4,ndF,type='response')
lines(ndF$dose,bw.respF*20,type='l',col='red')
points(rats[rats$sex=="F","dose"],rats[rats$sex=="F","numdead"],pch=19,
       col='red')


