import numpy as np

def score(predictions, y_test, salesPrice):
  # predictions,y_test,salesPrice array with equal length and value (order/Price) sorted by itemID ascending

  pred_sales = 0
  pred_cost = 0
  test_sales = 0


  for i in range(0, len(predictions), 1):

    if (predictions[i] <= y_test[i]): # without overstocking fee
      pred_sales += predictions[i] * salesPrice[i]

    else: # with overstocking fee
      pred_sales += y_test[i] * salesPrice[i]
      pred_cost += ((predictions[i]-y_test[i]) * salesPrice[i] * 0.6)

    test_sales += y_test[i] * salesPrice[i] # test orders

  return pred_sales, pred_cost, test_sales
