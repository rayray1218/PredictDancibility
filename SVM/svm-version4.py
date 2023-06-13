import pandas as pd
import numpy as np
import time
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV


#training data
train = pd.read_csv('train.csv')

y_train = train['Danceability']
x_train = train.drop(columns='Danceability')
x_train = x_train.drop(columns=['Uri', 'Url_spotify', 'Url_youtube',
                                 'id', 'Track', 'Album', 'Description', 'Title', 'Channel', 'Composer', 'Artist'])

#Danceability,Energy,Key,Loudness,Speechiness,Acousticness,Instrumentalness,Liveness,Valence,Tempo,Duration_ms,Views,Likes,Stream,Album_type,Licensed,official_video,id,Track,Album,Uri,Url_spotify,Url_youtube,Comments,Description,Title,Channel,Composer,Artist
for col in x_train.columns[0:9]:
    if col == x_train.columns[1]:
        mean = np.round(x_train[col].mean())
    else:
        mean = x_train[col].mean()
    x_train[col].fillna(mean, inplace=True)

for col in x_train.columns[9:13]:
    mean = np.round(x_train[col].mean())
    x_train[col].fillna(mean, inplace=True)

mean = np.round(x_train[x_train.columns[16]].mean())
x_train[x_train.columns[16]].fillna(mean, inplace=True)

#fill in blank in Album_type column with unknown
album_type_column = 'Album_type'
x_train[album_type_column].fillna('Unknown', inplace=True)
frequency_map = x_train[album_type_column].value_counts(normalize=True)
x_train[album_type_column] = x_train[album_type_column].map(frequency_map)



boolean_columns = ['Licensed', 'official_video']
#fill in blank in boolean colume with false
for col in boolean_columns:
    x_train[col].fillna(False, inplace=True)
    x_train[col] = x_train[col].astype(int)


x_train.to_csv('x_train_for_randomforest.csv', index=False)
#分為測試集跟訓練集
xtrain,xtest,ytrain,ytest=train_test_split(x_train,y_train,test_size=0.1,random_state=42)
#standardization

x_train = StandardScaler().fit_transform(xtrain)
x_intest=StandardScaler().fit_transform(xtest)

y_intest,y_train = ytest,ytrain


#test data
test = pd.read_csv('test.csv')
x_test = test.drop(columns=['Uri', 'Url_spotify', 'Url_youtube',
                                 'id', 'Track', 'Album', 'Description', 'Title', 'Channel', 'Composer', 'Artist'])

for col in x_test.columns[0:9]:
    if col == x_test.columns[1]:
        mean = np.round(x_test[col].mean())
    else:
        mean = x_test[col].mean()
    x_test[col].fillna(mean, inplace=True)

for col in x_test.columns[9:13]:
    mean = np.round(x_test[col].mean())
    x_test[col].fillna(mean, inplace=True)

mean = np.round(x_test[x_test.columns[16]].mean())
x_test[x_test.columns[16]].fillna(mean, inplace=True)

album_type_column = 'Album_type'
x_test[album_type_column].fillna('Unknown', inplace=True)
frequency_map = x_test[album_type_column].value_counts(normalize=True)
x_test[album_type_column] = x_test[album_type_column].map(frequency_map)


boolean_columns = ['Licensed', 'official_video']

for col in boolean_columns:
    x_test[col].fillna(False, inplace=True)
    x_test[col] = x_test[col].astype(int)

x_test.to_csv('x_test_for_randomforest.csv', index=False)

#standardization
scaler = StandardScaler()
X_test_scaled = scaler.fit_transform(x_test)
x_test = X_test_scaled

#True test data
truetest = pd.read_csv('test_partial_answer.csv')

y_true_test = truetest['Danceability']

#model select and perform V-fold cross-validation plus grid search in rbf and linear 
model = SVR()
param_grid = [
    {'kernel': [ 'rbf'],'C': [0.1,0.6,1,1.4,10],'gamma':[0.1,0.01,1,10,0.001]},]
#MAE乘以-1因為default是取最大值
num_folds = 5
grid_search = GridSearchCV(model, param_grid, cv=num_folds, scoring='neg_mean_absolute_error',refit=True)


#model training 
startTime = time.time()
grid_search.fit(x_train, y_train)
endTime = time.time()
print("Total Training Time:", endTime-startTime)
#print the best combinations
print("Best Parameters:", grid_search.best_params_)
print("Best Score (neg MAE):", -grid_search.best_score_)
cv_results = grid_search.cv_results_
mean_absolute_errors = -cv_results['mean_test_score']
best_model = grid_search.best_estimator_

#validation完之後的測試集
y_pred_intest=best_model.predict(x_intest)
y_pred_intest= np.round(y_pred_intest)
mae=mean_absolute_error(y_intest ,y_pred_intest)
print("Eout(split)",mae)
#predict
y_pred = best_model.predict(x_test)
y_pred= np.round(y_pred)
results = pd.DataFrame({'id': test['id'], 'Danceability': y_pred})

results.to_csv('svm-v4.csv', index=False)