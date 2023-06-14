import pandas as pd
import numpy as np
import time
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline

# training data
train = pd.read_csv('train.csv')

y_train = train['Danceability']
x_train = train.drop(columns='Danceability')
x_train = x_train.drop(columns=['Uri', 'Url_spotify', 'Url_youtube',
                                 'id', 'Track', 'Album', 'Description', 'Title', 'Channel', 'Composer', 'Artist'])


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
encoded_cols = pd.get_dummies(x_train[album_type_column], prefix=album_type_column, drop_first=True)
x_train = pd.concat([x_train, encoded_cols], axis=1)
x_train.drop(album_type_column, axis=1, inplace=True)


boolean_columns = ['Licensed', 'official_video', 'Album_type_album', 'Album_type_compilation', 'Album_type_single']
#fill in blank in boolean colume with false
for col in boolean_columns:
    x_train[col].fillna(False, inplace=True)
    x_train[col] = x_train[col].astype(int)


x_train.to_csv('x_train_for_LogReg.csv', index=False)

#testing
#x_train.drop(columns=['Licensed', 'official_video', 'Album_type_album', 'Album_type_compilation', 'Album_type_single'])

# #standardization
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(x_train)

# x_train = X_train_scaled
# y_train = y_train.values



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
encoded_cols = pd.get_dummies(x_test[album_type_column], prefix=album_type_column, drop_first=True)
x_test = pd.concat([x_test, encoded_cols], axis=1)
x_test.drop(album_type_column, axis=1, inplace=True)


boolean_columns = ['Licensed', 'official_video', 'Album_type_album', 'Album_type_compilation', 'Album_type_single']

for col in boolean_columns:
    x_test[col].fillna(False, inplace=True)
    x_test[col] = x_test[col].astype(int)


x_test.to_csv('x_test_for_LogReg.csv', index=False)

# #standardization
# scaler = StandardScaler()
# X_test_scaled = scaler.fit_transform(x_test)

# x_test = X_test_scaled


#model
model = make_pipeline(
    StandardScaler(),
    PolynomialFeatures(degree=1, include_bias=False),  # Specify the degree of polynomial features
    SGDRegressor(penalty='l2', alpha=0.01, max_iter=10000, tol=0.00001, random_state=721, learning_rate='constant',
                      eta0=0.0001, early_stopping=False)
    # SGDRegressor()
)

# Perform cross-validation
scores = cross_val_score(model, x_train, y_train, cv=5, scoring='neg_mean_absolute_error')
mae_scores = -scores  # Convert negative scores to positive
print(mae_scores)


# model training 
startTime = time.time()
model.fit(x_train, y_train)
endTime = time.time()
print("Total Training Time:", endTime-startTime)

# # # print(model.coef_)

#predict
y_pred = model.predict(x_test)
# y_pred = model.predict(x_test)
results = pd.DataFrame({'id': test['id'], 'Danceability': np.round(y_pred)})

results.to_csv('SGDRegressor.csv', index=False)


#Ein
y_pred = model.predict(x_train)
print(np.round(y_pred))
mae = mean_absolute_error(y_train, np.round(y_pred))
print(mae)