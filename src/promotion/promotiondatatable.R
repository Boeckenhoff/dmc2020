library(rprojroot)
library(lubridate)
library(data.table)
root = find_root(is_git_root)
orders = fread(file.path(root, "task/data/orders.csv"))

promotion = function(id){
  orders.id = orders[itemID == id]
  if(nrow(orders.id) > 2){
    orders.id[, date := as_date(time)]
    orders.id[, timediff := c(100, difftime(time[2:.N], time[1:(.N-1)], unit = "hours"))]
    tab = orders.id[,.(
      timediff = median(timediff),
      transactions = .N
    ), by = date]
    tab[, transactions.rel := transactions/sum(transactions)]
    tab[, promotion.index := timediff <= 0.1 & transactions.rel > 0.01]
    tab[, itemID := id]
    res = tab[promotion.index == TRUE, c("itemID", "date")]
  }else{
    res = data.table(itemID = integer(0), date = as_date(character(0)))
  }
  res
}

promotion.dates = rbindlist(lapply(sort(unique(orders$itemID)), promotion))
# Dauer < 7 Minuten

fwrite(promotion.dates, file = "team_c/features/promotion/promotiondates.csv")
