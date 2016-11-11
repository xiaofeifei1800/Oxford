
library(ggplot2)
tracking = read.table("P:/R Data/tracklong.txt", header = T)

#1.Perform an exploratory analysis of the data and summarise your findings. You may
#wish to consider some numerical summaries as well as some exploratory plots.
summary(tracking)
str(tracking)

# time_by_trial = split(tracking$time, tracking$trial)
# boxplot(time_by_trial, col = "aquamarine")
# 
# ggplot(aes(y = time, x = factor(trial),fill = shape), data = tracking) + geom_boxplot()+
#   facet_wrap(~ sex)

# with(tracking,
#      table(sex, cut(age, c(0, 20, 60), labels = c("YOUNG", "OLD")))
# )

plot(time[sex == "M"] ~ age[sex == "M"], data = tracking, pch = 16,
     ylab = "Time measured in trial 1 (seconds)", xlab = "Age (years)")
points(time[sex == "F"] ~ age[sex == "F"], data = tracking, pch = 16, col = 2)
abline(v = 20, lty = 2)
legend("topright", col = c(2, 1), pch = 16, legend = c("female", "male"))

par(mar = c(5, 6, 3, 3) + 0.1)
boxplot(time ~ trial + sex + shape, data = tracking,
        horizontal = TRUE, xlim = c(0, 17), ylim = c(0, 12), las = 1,
        xlab = "Value of time")

# smoothingSpline = smooth.spline(tracking$age, tracking$time, spar=0.35)
# plot(tracking$age, tracking$time)
# lines(smoothingSpline)

# 2.Model the relation between contact time and the trial, the tracker shape, and the
# subject’s age and sex using a normal linear model for the response
# Time, or some function of Time.You should consider possible interactions between the 
# explanatory variables. Carry out model selection and outlier analysis.

full_model = lm(time ~ sex * age * trial * shape, data = tracking)
stp = stepAIC(full_model, direction = "both")
#null_model = lm(time ~ 1, data = tracking)
#step(null_model, scope=list(lower=null_model, upper=full_model), direction="forward")

final = lm(formula = time ~ sex + age + trial + shape + sex:age + age:trial + age:shape, 
           data = tracking)

summary(final)
par(mfrow = c(2, 2))
plot(final)

boxcox(final,lambda = seq(-2, 2, length = 10))

new_tracking = tracking
new_tracking$time = (new_tracking$time)^(0.5)
temp = lm(time ~ sex * age * trial * shape, data = new_tracking)
stp = stepAIC(temp, direction = "both")
new_final = lm(formula = time ~ sex + age + trial + shape + sex:age + age:trial + age:shape, 
           data = new_tracking)
par(mfrow = c(2, 2))
plot(new_final)

#3.(a)Are the responses in the older(Age>20) and younger(Age<=20) subjects different?

tracking$grp <- factor(ifelse(tracking$age < 20, "young", "old"),
                      levels = c("young", "old"))

plot(tracking$grp,tracking$time)


#(b)Trial might  be  treated  as  a  categorical  variable  with  four  levels,  or  as  a
# continuous variable taking values 1, 2, 3 and 4. What are the relative merits of
# these model choices?

