#SVM模型

此模型為SVM之範例


#主要流程

一開始先將我們的模型選擇至SVM，然後利用V-fold validation 搭配grid search 去做我們的模型參數選擇，最後選擇完之後再利用最適合之模型去計算最後測試集的預測結果以及真實結果之MAE。