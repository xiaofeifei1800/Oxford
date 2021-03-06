########### tm
fuzzy_prep_words <- function(words) {
  # Prepares a list of words for fuzzy matching. All the other fuzzy matching
  # functions will run word through this. Given a list of sentences, returns
  # a list of words.
  
  words <- unlist(strsplit(tolower(gsub("[[:punct:]]", " ", words)), "\\W+"))
  return(words)
}

fuzzy_gen_word_freq <- function(l, fun = identity) {
  # Returns a word frequency vector based on vector of sentences l and with
  # frequencies post-processed by fun (e.g. log)
  
  fun(sort(table(fuzzy_prep_words(unlist(strsplit(l, ' ')))), decreasing=T))+1
}

fuzzy_title_match <- function(a, b, wf) {
  # Fuzzy matches a performance title based on a custom algorithm tuned for
  # this purpose. Words are frequency-weighted (like tf-idf).
  # 
  # Args:
  #   a, b: the two titles to match
  #   wf: a vector of word frequencies as generated by fuzzy_gen_word_freq
  #
  # Returns:
  #   A fuzzy match score, higher is better, +Inf for exact match
  
  if (a == b) # Shortcut to make faster
    return (Inf)
  a.words <- fuzzy_prep_words(a)
  b.words <- fuzzy_prep_words(b)
  a.freqs <- sapply(a.words, function(x) { ifelse(is.na(wf[x]), 1, wf[x]) })
  b.freqs <- sapply(b.words, function(x) { ifelse(is.na(wf[x]), 1, wf[x]) })
  
  d <- adist(a.words, b.words)
  a.matches <- 1-apply(d, 1, function(x) { min(x, 2) })/2
  b.matches <- 1-apply(d, 2, function(x) { min(x, 2) })/2
  
  matchsum <- min(sum(a.matches * 1/a.freqs), sum(b.matches * 1/b.freqs))
  unmatchsum <- sum(floor(1-a.matches) * 1/a.freqs) + sum(floor(1-b.matches) * 1/b.freqs)
  return(matchsum / unmatchsum)
}
# Example - outer function needs a vectorised function so there's a little extra work, otherwise this is pretty simple
# The scores matrix contains all the pairwise scores. Then it would be a simple matter to pick the best match for each
# with details depending on whether there can be multiple matches, whether everything must match, etc.

# choose for test set

ice = nutri[nutri$MicroDeptName == "Premium Ice Cream",]
ice = na.omit(ice)

sugar = nutri[nutri$MicroDeptName == "sugar bags",]
sugar = na.omit(sugar)

test_ice = sample(dim(ice)[1])[1:5]
test_ice = ice$SkuName[test_ice]

test_sugar = sample(dim(sugar)[1])[1:5]
test_sugar = sugar$SkuName[test_sugar]

ice = ice[!ice$SkuName %in% test_ice,]
sugar = sugar[!sugar$SkuName %in% test_sugar,]

SkuName = gsub("\\s*\\w*$", "", sugar$SkuName)
A = SkuName
B = test_sugar
wf <- fuzzy_gen_word_freq(c(A, B))
vectorised_match <- function (L1,L2) { mapply(function(a,b) { fuzzy_title_match(a, b, wf) }, L1, L2) } 
scores <- outer(A, B, vectorised_match)
rownames(scores) <- A
colnames(scores) <- B

df <- melt(as.matrix(scores), varnames = c("name1", "name2"))
df = df[!duplicated(df),]

library(dplyr)
library(data.table)

pred = df %>% group_by(name2) %>% filter(value == max(value))
sugar$SkuName = gsub("\\s*\\w*$", "", sugar$SkuName)
pred_result = sugar[sugar$SkuName%in%as.character(pred$name1),]

pred = as.data.table(pred)
pred = pred[pred$value>0.2]
pred_result = as.data.table(pred_result)
colnames(pred)[1] = "SkuName"
setkey(pred, SkuName)
setkey(pred_result, SkuName)
pred_result = merge(pred, pred_result, all.x=TRUE)

pred_result = pred_result[,c("name2", "Sugars per 100g", "Sugars per Portion")]
colnames(pred_result) = c("SkuName", "Sugars per 100_p", "Sugars per Portion_p")
sugar = nutri[nutri$MicroDeptName == "sugar bags",]
true_result = sugar[sugar$SkuName %in%pred_result$SkuName,]

setkey(pred_result, SkuName)
setkey(true_result, SkuName)
result = merge(true_result, pred_result, all.x=TRUE)

mse_100 = mean((result$`Sugars per 100g`- result$`Sugars per 100_p`)^2)
mse_por = mean((result$`Sugars per Portion`- result$`Sugars per Portion_p`)^2)

mean_100 = colMeans(sugar[!sugar$SkuName%in%result$SkuName, "Sugars per 100g"],na.rm = T)
mse_por = colMeans(sugar[!sugar$SkuName%in%result$SkuName, "Sugars per Portion"],na.rm = T)

mean_100 = as.numeric(mean_100)
mse_por = as.numeric(mse_por)

mse_h_100 = mean((result$`Sugars per 100g`- mean_100)^2)
mse_h_por = mean((result$`Sugars per Portion`- mse_por)^2)

#########
SkuName = gsub("\\s*\\w*$", "", ice$SkuName)
A = SkuName
B = test_ice
wf <- fuzzy_gen_word_freq(c(A, B))
vectorised_match <- function (L1,L2) { mapply(function(a,b) { fuzzy_title_match(a, b, wf) }, L1, L2) } 
scores <- outer(A, B, vectorised_match)
rownames(scores) <- A
colnames(scores) <- B

df <- melt(as.matrix(scores), varnames = c("name1", "name2"))
df = df[!duplicated(df),]

library(dplyr)
library(data.table)

pred = df %>% group_by(name2) %>% filter(value == max(value))
ice$SkuName = gsub("\\s*\\w*$", "", ice$SkuName)
pred_result = ice[ice$SkuName%in%as.character(pred$name1),]

pred = as.data.table(pred)
pred = pred[pred$value>0.2]
pred_result = as.data.table(pred_result)
colnames(pred)[1] = "SkuName"
setkey(pred, SkuName)
setkey(pred_result, SkuName)
pred_result = merge(pred, pred_result, all.x=TRUE)

pred_result = pred_result[,c("name2", "Sugars per 100g", "Sugars per Portion")]
colnames(pred_result) = c("SkuName", "Sugars per 100_p", "Sugars per Portion_p")
ice = nutri[nutri$MicroDeptName == "Premium Ice Cream",]
true_result = ice[ice$SkuName %in%pred_result$SkuName,]

setkey(pred_result, SkuName)
setkey(true_result, SkuName)
result = merge(true_result, pred_result, all.x=TRUE)

mse_100 = mean((result$`Sugars per 100g`- result$`Sugars per 100_p`)^2)
mse_por = mean((result$`Sugars per Portion`- result$`Sugars per Portion_p`)^2)

mean_100 = colMeans(ice[!ice$SkuName%in%result$SkuName, "Sugars per 100g"],na.rm = T)
mse_por = colMeans(ice[!ice$SkuName%in%result$SkuName, "Sugars per Portion"],na.rm = T)

mean_100 = as.numeric(mean_100)
mse_por = as.numeric(mse_por)

mse_h_100 = mean((result$`Sugars per 100g`- mean_100)^2)
mse_h_por = mean((result$`Sugars per Portion`- mse_por)^2)
