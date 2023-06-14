# 預測歌曲可跳性的模型

這個項目使用不同的模型來預測歌曲的Dancibility。我們使用一份資料集來設計和評估這些模型，並使用MAE（平均絕對誤差）作為評估指標，以找到越來越好的模型。

## 資料集

使用的資料集包含歌曲的特徵和對應的可跳性分數。這些特徵可以包括節奏、曲風、歌手等等。

## 模型

我們設計了以下不同的模型來進行可跳性預測：

1. SGD regression
2. SVM
3. CNN
4. Random Forest
