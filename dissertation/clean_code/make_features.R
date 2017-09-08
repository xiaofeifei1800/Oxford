library(data.table)

nutri = fread("/Users/xiaofeifei/I/Oxford/Dissertation/Nutrition data.csv")
focus_group = fread("/Users/xiaofeifei/I/Oxford/Dissertation/sub_g_data.csv")

main_sub = c("CEREAL", 'YOG', '311', '310', 'XBISC', 'LAY013', 'LAY083', 'DAY-SWEETS',
             'ICREAM', '335', '338', '390', '396', '45')

focus_group <- nutri[nutri$SubDeptCode %in% main_sub]
focus_group = na.omit(focus_group)
focus_group$SkuCode = as.character(focus_group$SkuCode)
focus_group[,code1 := substr(SkuCode, start = 1, stop = 4)]
focus_group[,code2 := substr(SkuCode, start = 5, stop = 8)]
focus_group[,code3 := substr(SkuCode, start = 9, stop = 13)]

changeCols = c("code1", "code2", "code3")
focus_group[,(changeCols):= lapply(.SD, as.numeric), .SDcols = changeCols]

focus_group[is.na(focus_group)] = 0

colnames(focus_group)[17] = "Sugars"
fwrite(focus_group, "/Users/xiaofeifei/I/Oxford/Dissertation/sub_g_data.csv", row.names = F)
