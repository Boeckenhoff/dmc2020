library(date)
library(data.table)
library(dplyr)
library(zoo)
# Daten im daily Format einlesen: die Spalten itemID, date, promotion, category1, category2, category3, manufacturer, brand werden benötigt.
root = rprojroot::find_root(rprojroot::is_git_root)
load(file.path(root, "team_c/DMC-2020-Task/DMC20_Data/data_promo.RData"))

#feature "similar product is in promotion" berechnet für jedes Item welche ähnlichen Produkte (gleicher Hersteller, Marke, Kategorien) am betrachteten Tag (alle Tage im Trainingszeitraum) in Promotion sind.
similarProductIsInPromotion=rep(FALSE, length(data_promo$itemID))
similarProductsInPromotionCount=rep(0, length(data_promo$itemID))
for (d in 1:180){ #180 Tage
  day<-data_promo$date[d]
  data_promoDay<-data_promo[date==day]
  similarProductInPromotionDay=rep(FALSE, length(data_promoDay$itemID))
  similarProductsInPromotionCountDay=rep(0,length(data_promoDay$itemID)) # Berechnet den Anteil ähnlicher Produkte in Promotion
  for (i in 1:length(data_promoDay$itemID)){
    similarProductsInPromotion<-data_promoDay[promotion==TRUE & category1==category1[i] & category2==category2[i] & category3==category3[i] & manufacturer==manufacturer[i] & brand==brand[i] & itemID!=itemID[i]]
    similarProductsInPromotionCountDay[i]<-nrow(similarProductsInPromotion)
    if (nrow(similarProductsInPromotion)!=0){
      similarProductInPromotionDay[i]=TRUE
    }
  }
  similarProductIsInPromotion[((d-1)*nrow(data_promoDay)+1):(d*nrow(data_promoDay))]= similarProductInPromotionDay[1:length(similarProductInPromotionDay)]
  similarProductsInPromotionCount[((d-1)*nrow(data_promoDay)+1):(d*nrow(data_promoDay))]= similarProductsInPromotionCountDay
  print(d)
}
data_promo<-data_promo[order(date, itemID)]
data_promo$similarProductIsInPromotion=similarProductIsInPromotion
data_promo$similarProductsInPromotionCount=similarProductsInPromotionCount

#Berechne für jedes Item die Anzahl ähnlicher Items
numberSimilarItems=rep(0,10463) #10463 Items
itemData<-data_promo[1:10463]
for (i in 1:10463){ 
  numberSimilarItems[i]<-nrow(itemData[category1==category1[i] & category2==category2[i] & category3==category3[i] & manufacturer==manufacturer[i] & brand==brand[i] & itemID!=itemID[i]])
  }
#Berechne für jedes Item den prozentualen Anteil aehnlicher Produkte in Promotion 
numberSimilarItems<-rep.int(numberSimilarItems,180)
similarProductsInPromotionProportion<-(similarProductsInPromotionCount/numberSimilarItems)*100
is.na(similarProductsInPromotionProportion) <- 0
data_promo$similarProductsInPromotionProportion=similarProductsInPromotionProportion

data_promo<-data_promo[order(itemID,date)]
data_promo$similarProductsInPromotionProportion[is.na(data_promo$similarProductsInPromotionProportion)] <- 0
#save(data_promo, file="data_promo_SPIIP.RData")
featureSPIIP<-select(data_promo, itemID, date, similarProductIsInPromotion, similarProductsInPromotionCount, similarProductsInPromotionProportion)
#write.csv(featureSPIIP, file="featureSPIIP.csv", row.names = FALSE)
#save(featureSPIIP, file="featureSPIIP.RData")




