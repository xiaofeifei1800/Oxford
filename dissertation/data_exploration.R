library(data.table)
weights = fread("/Users/xiaofeifei/I/Oxford/Dissertation/Nutrition weights data.csv")
sales = fread("/Users/xiaofeifei/I/Oxford/Dissertation/Nutrition sales data.csv")

# combine data
setkey(weights, SkuCode)
setkey(sales, SkuCode)
nutri = merge(sales,weights[,SkuName:=NULL],all.x = T)
rm(sales, weights)

# clean data
# change to numeric
str(data)
changeCols = c("SkuCode", "SalesUnits ", "Energy per portion", "Sugars per Portion", "Fat per Portion",
               "Saturates per Portion", "Salt per Portion")
data[,(changeCols):= lapply(.SD, as.numeric), .SDcols = changeCols]
str(data)

# basic start exploration
library(plotly)
nutri = as.data.frame(nutri)
p1 <- plot_ly(nutri, x = ~DeptCode)
p1
p2 <- plot_ly(nutri, x = ~SubDeptCode)
p2

one_plot <- function(d) {
  plot_ly(d, x = ~price) %>%
    add_annotations(
      ~unique(clarity), x = 0.5, y = 1, 
      xref = "paper", yref = "paper", showarrow = FALSE
    )
}

diamonds %>%
  split(.$clarity) %>%
  lapply(one_plot) %>% 
  subplot(nrows = 2, shareX = TRUE, titleX = FALSE) %>%
  hide_legend()