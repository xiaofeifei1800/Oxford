library(data.table)
nutri = fread("/Users/xiaofeifei/I/Oxford/Dissertation/Nutrition data.csv")
# separate 100g and portion
nutri_portion = c(colnames(nutri)[1:10],"Energy per portion", "Sugars per Portion", "Fat per Portion",
                  "Saturates per Portion", "Salt per Portion")

nutri_portion = nutri[,nutri_portion,with=F]

nutri_100g = c(colnames(nutri)[1:10],"Energy per 100g", "Sugars per 100g", "Fat per 100g",
               "Saturates per 100g", "Salt per 100g")

nutri_100g = nutri[,nutri_100g,,with=F]

rm(nutri)
###########################
# basic start exploration
##########################
#portion of each variables
summary(nutri_portion)
summary(nutri_100g)
apply(nutri_portion[,11:15], 2, var,na.rm=T)
apply(nutri_100g[,11:15], 2, var,na.rm=T)

nutri_portion[, sum := rowSums(nutri_portion[,11:15], na.rm = T)]
nutri_100g[, sum := rowSums(nutri_100g[,11:15], na.rm = T)]

apply(na.omit(nutri_portion[,11:15]/nutri_portion$sum),2,mean)
apply(na.omit(nutri_100g[,11:15]/nutri_100g$sum),2,mean)

#portion of each variables by dept
by(nutri_portion, nutri_portion$DeptCode, function(x) 
  apply(na.omit(x[,11:15]/x$sum),2,mean))

by(nutri_100g, nutri_100g$DeptCode, function(x) 
  apply(na.omit(x[,11:15]/x$sum),2,mean))


######################
library(ggplot2)
nutri = as.data.frame(nutri)
a = as.data.frame(table(nutri$DeptCode))
barplot(a$Freq,
        main = "Cound number of each Department",
        xlab = "Counts",
        ylab = "DeptCode",
        names.arg = a$Var1,
        col = "darkred",
        horiz = TRUE)

a = as.data.frame(table(nutri$SubDeptCode))
barplot(a$Freq,
        main = "Cound number of each Subdepartment",
        xlab = "Counts",
        ylab = "SubdeptCode",
        names.arg = a$Var1,
        col = "darkred",
        horiz = TRUE)
rm(a)

nutri = as.data.frame(nutri)
num = colnames(nutri)[11:20]
mean_by_dept = aggregate(nutri[,num], list(nutri$DeptCode), mean, na.rm=T)

mean_by_dept = as.data.table(mean_by_dept)
name = mean_by_dept$Group.1
mean_by_dept[,Group.1:=NULL]
portion = mean_by_dept[,c("Energy per portion", "Sugars per Portion", "Fat per Portion",
                          "Saturates per Portion", "Salt per Portion")]
barplot(t(portion), beside = T, names.arg = name, cex.names=0.6, col= c(1,2,3,4,5))
legend("topright",
       colnames(portion),
       fill = c(1,2,3,4,5)
)

g_100 = mean_by_dept[,!c("Energy per portion", "Sugars per Portion", "Fat per Portion",
                         "Saturates per Portion", "Salt per Portion")]
barplot(t(g_100), beside = T, names.arg = name, cex.names=0.6, col= c(1,2,3,4,5))
legend("topright",
       colnames(g_100),
       fill = c(1,2,3,4,5)
)

hist(nutri$`Energy per 100g`, )
hist(nutri$`Fat per 100g`)
hist(nutri$`Saturates per 100g`)
hist(nutri$`Sugars per 100g`)
hist(nutri$`Salt per 100g`)

hist(nutri$`Energy per portion`)
hist(nutri$`Fat per Portion`)
hist(nutri$`Saturates per Portion`)
hist(nutri$`Sugars per Portion`)
hist(nutri$`Salt per Portion`)

hist(nutri[nutri$DeptCode=="HSP","Energy per portion"])

# plot densities 
library(sm)
remove_na = na.omit(nutri)
sm.density.compare(remove_na$`Energy per 100g`, as.factor(remove_na$DeptCode))
title(main="Energy per 100g by DeptCode")
colfill<-c(2:(2+length(levels(as.factor(remove_na$DeptCode))))) 
legend("topright", levels(as.factor(remove_na$DeptCode)), fill=colfill)

sm.density.compare(remove_na$`Energy per portion`, as.factor(remove_na$DeptCode))
title(main="Energy per portion by DeptCode")
colfill<-c(2:(2+length(levels(as.factor(remove_na$DeptCode))))) 
legend("topright", levels(as.factor(remove_na$DeptCode)), fill=colfill)

sm.density.compare(remove_na$`Fat per 100g`, as.factor(remove_na$DeptCode))
title(main="Fat per 100g by DeptCode")
colfill<-c(2:(2+length(levels(as.factor(remove_na$DeptCode))))) 
legend("topright", levels(as.factor(remove_na$DeptCode)), fill=colfill)

sm.density.compare(remove_na$`Fat per Portion`, as.factor(remove_na$DeptCode))
title(main="Fat per Portion by DeptCode")
colfill<-c(2:(2+length(levels(as.factor(remove_na$DeptCode))))) 
legend("topright", levels(as.factor(remove_na$DeptCode)), fill=colfill)

sm.density.compare(remove_na$`Sugars per 100g`, as.factor(remove_na$DeptCode))
title(main="Sugars per 100g by DeptCode")
colfill<-c(2:(2+length(levels(as.factor(remove_na$DeptCode))))) 
legend("topright", levels(as.factor(remove_na$DeptCode)), fill=colfill)

sm.density.compare(remove_na$`Sugars per Portion`, as.factor(remove_na$DeptCode))
title(main="Sugars per Portion by DeptCode")
colfill<-c(2:(2+length(levels(as.factor(remove_na$DeptCode))))) 
legend("topright", levels(as.factor(remove_na$DeptCode)), fill=colfill)

# corelation suger with other varible
cor(na.omit(nutri_portion)[,11:15])

nutri_portion[,energy_fat := `Energy per portion`- `Fat per Portion`]
cor(scale(na.omit(nutri_portion)[,11:16]))

