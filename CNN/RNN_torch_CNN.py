import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# 處理數值特徵
numeric_features = ['Energy', 'Key', 'Loudness', 'Speechiness', 'Acousticness',
                    'Instrumentalness', 'Liveness', 'Valence', 'Tempo', 'Duration_ms',
                    'Views', 'Likes', 'Stream']  # 13維數值特徵

data = pd.read_csv('train.csv')

# 分割特徵和目標變數
X = data.drop('Danceability', axis=1)
y = data['Danceability']
X = X.drop(['id', 'Uri', 'Url_spotify', 'Url_youtube'], axis=1)

for col in numeric_features:
    mean_val = X[col].mean()
    X[col].fillna(mean_val, inplace=True)
X_numeric = X[numeric_features].values

scaler = StandardScaler()
X_numeric_scaled = scaler.fit_transform(X_numeric)

# 處理布林值特徵
boolean_features = ['Licensed', 'official_video']  # 2維布林值特徵

for col in ['Licensed', 'official_video']:
    X[col] = (1 * X[col])
    X[col].fillna(0.5, inplace=True)

X_boolean = X[boolean_features].values

# 處理字串特徵
categorical_feature = 'Album_type'  # 分類特徵，只會出現 'album', 'single', 'compilation' 和 NA
X_categorical = X[categorical_feature].fillna('NA').values

label_encoder = LabelEncoder()
X_categorical_encoded = label_encoder.fit_transform(X_categorical)

max_sequence_length = 100  # 最大序列長度
max_num_words = 500  # 最大詞彙數量

# 處理普通字串特徵
string_features = ['Track', 'Album', 'Title', 'Channel', 'Composer', 'Artist']  # 6維普通字串特徵
X_string = X[string_features].fillna('NA').values

tokenizer = Tokenizer(num_words=max_num_words)
tokenizer.fit_on_texts(X_string.tolist())

X_string_sequences = tokenizer.texts_to_sequences(X_string.tolist())
X_string_padded = pad_sequences(X_string_sequences, maxlen=max_sequence_length)

# 處理長段文字敘述特徵
text_feature = 'Description'  # 長段文字敘述特徵
X_text = X[text_feature].fillna('').values

tokenizer = Tokenizer(num_words=max_num_words)
tokenizer.fit_on_texts(X_text.tolist())

X_text_sequences = tokenizer.texts_to_sequences(X_text.tolist())
X_text_padded = pad_sequences(X_text_sequences, maxlen=max_sequence_length)

# 分割訓練集和測試集
X_train_numeric, X_val_numeric, X_train_boolean, X_val_boolean, X_train_categorical, X_val_categorical, X_train_string, X_val_string, X_train_text, X_val_text, y_train, y_val = train_test_split(
    X_numeric_scaled, X_boolean, X_categorical_encoded, X_string_padded, X_text_padded, y, test_size=0.1, random_state=42
)

# 轉換為TensorDataset
train_dataset = TensorDataset(
    torch.tensor(X_train_numeric, dtype=torch.float32),
    torch.tensor(X_train_boolean, dtype=torch.float32),
    torch.tensor(X_train_categorical, dtype=torch.long),
    torch.tensor(X_train_string, dtype=torch.long),
    torch.tensor(X_train_text, dtype=torch.long),
    torch.tensor(y_train.values, dtype=torch.float32)
)
val_dataset = TensorDataset(
    torch.tensor(X_val_numeric, dtype=torch.float32),
    torch.tensor(X_val_boolean, dtype=torch.float32),
    torch.tensor(X_val_categorical, dtype=torch.long),
    torch.tensor(X_val_string, dtype=torch.long),
    torch.tensor(X_val_text, dtype=torch.long),
    torch.tensor(y_val.values, dtype=torch.float32)
)

# 建立模型
class MyModel(nn.Module):
    def __init__(self, numeric_dim, boolean_dim, categorical_dim, string_dim, text_dim, embedding_dim, num_filters):
        super(MyModel, self).__init__()
        self.numeric_layer = nn.Linear(numeric_dim, 64)
        self.boolean_layer = nn.Linear(boolean_dim, 32)
        self.categorical_layer = nn.Embedding(categorical_dim, 10)
        self.string_layer = nn.Embedding(string_dim, embedding_dim)
        self.text_layer = nn.Embedding(text_dim, embedding_dim)
        self.conv1d = nn.Conv1d(embedding_dim, num_filters, kernel_size=5)
        self.flatten = nn.Flatten()
        # self.cnn = nn.Sequential( #1.7493 1.8335 no unsqueeze
        #     nn.Linear((64 + 32 + 10 + 32 + 32), 64),
        #     nn.ReLU(),
        #     nn.Linear(64, 64),
        #     nn.ReLU()
        # )
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * (64 + 32 + 10 + 32 + 32 - 2), 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU()
        )
        self.output_layer = nn.Linear(64, 1)

    def forward(self, numeric, boolean, categorical, string, text):
        numeric_out = self.numeric_layer(numeric)
        boolean_out = self.boolean_layer(boolean)
        categorical_out = self.categorical_layer(categorical).flatten(start_dim=1)
        # print(categorical_out.size()) #(128, 10)
        string_out = self.string_layer(string)
        string_out = self.conv1d(string_out.transpose(1, 2)).max(dim=2)[0]
        # print(string_out.size()) #(128, 32)
        text_out = self.text_layer(text)
        text_out = self.conv1d(text_out.transpose(1, 2)).max(dim=2)[0]
        # print(text_out.size()) #(128, 32)
        flattened = self.flatten(torch.cat((numeric_out, boolean_out, categorical_out, string_out, text_out), dim=1))
        flattened = flattened.unsqueeze(1)
        dense_out = self.cnn(flattened)
        output = self.output_layer(dense_out)
        return output

# 建立模型
numeric_dim = 13
boolean_dim = 2
categorical_dim = 4
string_dim = max_num_words
text_dim = max_num_words
embedding_dim = 10
num_filters = 32

model = MyModel(numeric_dim, boolean_dim, categorical_dim, string_dim, text_dim, embedding_dim, num_filters)

# 設定訓練參數
criterion = nn.L1Loss()

# 設定批次大小和資料載入器
batch_size = 256 # 128:1.8662 64:1.8655 256: 1.8921
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

# 訓練模型
epochs = 500
for epoch in range(epochs):
    model.train()
    # print(0.0005*(1 - (epoch / (2 * epochs))))0.002*max(0.05, (1 - (epoch / epochs)))
    optimizer = optim.SGD(model.parameters(), lr=0.0005*(1 - (epoch / (2 * epochs))))
    train_loss = 0.0
    for inputs_numeric, inputs_boolean, inputs_categorical, inputs_string, inputs_text, targets in train_dataloader:
        optimizer.zero_grad()
        outputs = model(inputs_numeric, inputs_boolean, inputs_categorical, inputs_string, inputs_text)
        loss = criterion(outputs.squeeze(), targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * inputs_numeric.size(0)
    train_loss /= len(train_dataset)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs_numeric, inputs_boolean, inputs_categorical, inputs_string, inputs_text, targets in val_dataloader:
            outputs = model(inputs_numeric, inputs_boolean, inputs_categorical, inputs_string, inputs_text)
            loss = criterion(outputs.squeeze(), targets)
            val_loss += loss.item() * inputs_numeric.size(0)
        val_loss /= len(val_dataset)

    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

model.eval()
data_test = pd.read_csv('test.csv')

# 分割特徵和目標變數
X_test = data_test.drop(['id', 'Uri', 'Url_spotify', 'Url_youtube'], axis=1)

for col in numeric_features:
    mean_val = X_test[col].mean()
    X_test[col].fillna(mean_val, inplace=True)
X_numeric = X_test[numeric_features].values

X_numeric_scaled = scaler.fit_transform(X_numeric)

for col in ['Licensed', 'official_video']:
    X_test[col] = (1 * X_test[col])
    X_test[col].fillna(0.5, inplace=True)

X_boolean = X_test[boolean_features].values

X_categorical = X_test[categorical_feature].fillna('NA').values

label_encoder = LabelEncoder()
X_categorical_encoded = label_encoder.fit_transform(X_categorical)

X_string = X_test[string_features].fillna('NA').values

tokenizer = Tokenizer(num_words=max_num_words)
tokenizer.fit_on_texts(X_string.tolist())

X_string_sequences = tokenizer.texts_to_sequences(X_string.tolist())
X_string_padded = pad_sequences(X_string_sequences, maxlen=max_sequence_length)

X_text = X_test[text_feature].fillna('').values

tokenizer = Tokenizer(num_words=max_num_words)
tokenizer.fit_on_texts(X_text.tolist())

X_text_sequences = tokenizer.texts_to_sequences(X_text.tolist())
X_text_padded = pad_sequences(X_text_sequences, maxlen=max_sequence_length)

# 分割訓練集和測試集
# X_train_numeric, X_val_numeric, X_train_boolean, X_val_boolean, X_train_categorical, X_val_categorical, X_train_string, X_val_string, X_train_text, X_val_text, y_train, y_val = train_test_split(
#     X_numeric_scaled, X_boolean, X_categorical_encoded, X_string_padded, X_text_padded, y, test_size=0.2, random_state=42
# )

# 轉換為TensorDataset
test_dataset = TensorDataset(
    torch.tensor(X_numeric_scaled, dtype=torch.float32),
    torch.tensor(X_boolean, dtype=torch.float32),
    torch.tensor(X_categorical_encoded, dtype=torch.long),
    torch.tensor(X_string_padded, dtype=torch.long),
    torch.tensor(X_text_padded, dtype=torch.long)
)

test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

y_pred = []
with torch.no_grad():
    for inputs_numeric, inputs_boolean, inputs_categorical, inputs_string, inputs_text in test_dataloader:
        outputs = model(inputs_numeric, inputs_boolean, inputs_categorical, inputs_string, inputs_text)
        # print(type(outputs))
        # print(outputs.size())
        y_pred.extend(outputs.numpy().tolist())

y_pred = np.array(y_pred)
y_pred = np.round(y_pred.flatten())  # 四捨五入為整數
y_pred[y_pred < 0] = 0
y_pred[y_pred > 9] = 9
results = pd.DataFrame({'id': data_test['id'], 'Danceability': y_pred})
# 寫入CSV
results.to_csv('RNN_torch_CNN.csv', index=False)