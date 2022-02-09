library(date)
library("data.table")
root = rprojroot::find_root(rprojroot::is_git_root)
orders = fread(file.path(root, "task/data/orders.csv"))
orders_daily <- fread(file.path(root, "team_c/DMC-2020-Task/DMC20_Data/daily.csv")) #Datei "daily.csv" einlesen (von Gruppe A bereitgestellt)

promotion <- function(ID, orders) #ID: itemID
{
  itemIDIndex <- orders$itemID==ID
  orders.ID<- orders[itemIDIndex,] #Alle Kaeufe des Produkts mit der eingegebenen ID 
  if(length(orders.ID$time)>2){
    timeVector1<- orders.ID$time[1:length(orders.ID$time)-1] 
    timeVector2<- orders.ID$time[2:length(orders.ID$time)]
    diffTimeVector <- matrix(, nrow = length(orders.ID$time), ncol = 1)
    diffTimeVector[2:length(orders.ID$time),] = as.numeric(difftime(timeVector2,timeVector1,unit="hours" )) # zeitliche Differenz (in Stunden) zwischen zwei Transaktionen
    
    
    orders.ID$date <- as.Date(orders.ID$time, "%Y-%m-%d")
    diffTimeVector[1,] = 100 #hoher Wert (beliebig gewaehlt)
    orders.ID$diffTime <- diffTimeVector
    orders.ID.timediff.per.day <- aggregate(orders.ID$diffTime, by=list(date = orders.ID$date), median) # ueber den Tag gemittelte (Median) zeitliche Differenz zwischen zwei Transaktionen
    orders.ID.sumTransactions.per.day <- aggregate(rep(1,length(orders.ID$date)), by=list(date = orders.ID$date), sum) # Anzahl der Transaktionen pro Tag
    
    promotion.index<- orders.ID.timediff.per.day[,2]<=0.1 & orders.ID.sumTransactions.per.day[,2]/length(orders.ID$time)>0.01 # Deklaration als Promotion, falls der gemittelte Abstand der Transaktionen kleiner 0,1 h und mindestens 1% der Produkte in diesem Zeitraum gekauft werden. 
    promotion.ID <- orders.ID.timediff.per.day[promotion.index,]
  }
  else{
    promotion.ID <- data.frame(NULL, as.Date(x = integer(0), origin = "1970-01-01"))
  }
  return(promotion.ID[,1])
}

newList <- vector(mode = "list", length = 10463)

N <- 10463
for(i in c(1:N)){
  resultList <- promotion(i,orders)
  newList[[i]] <- resultList
}

save(newList,file="promotionList.RData")



 #Fuegt die Spalte promotion (boolean) an orders hinzu. promotion = TRUE gdw. das Item an dem Tag der Transaktion in Promotion war
 #input  promotionDays: Eine Liste von Liste mit den Promotion Days der jeweiligen Items. 
 #         wichtig: die Liste promotionDays[[i]] gehoert zu dem Item mit der ID i
 #       data orders: time | transactID | itemID | order | salesPrice
 #output: new data: time | transactID | itemID | order | salesPrice | promotion       
 addPromotionFeature <- function(promotionDays, orders){
   numberOfEntries <- length(orders$itemID)
   promotion <- rep(FALSE, numberOfEntries)
   
   #ueberprueft fuer jeden Eintrag, ob das item an dem Tag der Transaktion in Promotion war
   for (i in c(1:numberOfEntries)) {
     ID <- orders$itemID[i]
     promotionDaysOfID <- promotionDays[[ID]] #Liste der Promotionstage des Items aus der betrachteten Transaktion
     promotion[i] <- as.IDate(orders$date[i]) %in% as.IDate(promotionDaysOfID, "%Y-%m-%d") #Ist der Transaktionstag in der Liste der Promotionstage?
   }
   
   return( cbind(orders, promotion))
 }

ordersPromotion <- addPromotionFeature(newList, orders_daily) # Promotion feature hinzufuegen
save(ordersPromotion, file="ordersPromotion.RData")
promotion <-ordersPromotion[, .(itemID, date, promotion)]
write.csv(promotion, file="promotion.csv", row.names = FALSE)

