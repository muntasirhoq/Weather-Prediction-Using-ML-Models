import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('Data/raw_data.csv')
df_bbq = pd.read_csv('Data/labels.csv')
df.head()

df_Dresden = df[['DATE', 'DRESDEN_cloud_cover', 'DRESDEN_wind_speed', 'DRESDEN_wind_gust',
       'DRESDEN_humidity', 'DRESDEN_global_radiation', 'DRESDEN_precipitation',
       'DRESDEN_sunshine', 'DRESDEN_temp_mean', 'DRESDEN_temp_min',
       'DRESDEN_temp_max']]

df_Dresden['BBQ'] = df_bbq['DRESDEN_BBQ_weather']
print(df_Dresden["BBQ"].value_counts())
df_Dresden.head()

df_Dresden['BBQ'] = df_Dresden['BBQ'].map({True:1, False:0})

df_Dresden.head()

df_Dresden.isna().sum()

X = df_Dresden.drop(['DATE','BBQ'], axis = 1)
y = df_Dresden['BBQ']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 4)
print(len(X_train), len(X_test))
print(X_train.shape)

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report,confusion_matrix, accuracy_score

# Create a decision tree classifier
clf = DecisionTreeClassifier(random_state=4)

# Fit the classifier to the training data
clf.fit(X_train, y_train)

# Predict on the test data
y_pred = clf.predict(X_test)

# Print the classification report
report = classification_report(y_test, y_pred)
print(report)

# Retrieve feature importances
feature_importance = clf.feature_importances_

# Mapping feature names to their importance scores
feature_names = X.columns  # Assuming X is a DataFrame and contains feature names
feature_importance_map = dict(zip(feature_names, feature_importance))

# Sort feature importance in descending order
sorted_feature_importance = sorted(feature_importance_map.items(), key=lambda x: x[1], reverse=True)

# Print feature importance
print("Feature Importance:")
for feature, importance in sorted_feature_importance:
    print(f"{feature}: {importance}")
    
from keras.models import Model
from keras.layers import Input, Conv2D, Dense, concatenate, Embedding, Concatenate, Dropout, TimeDistributed, Softmax
from tensorflow import keras
import tensorflow as tf
from keras import callbacks
import tensorflow.keras.backend as K


# feature_size: the vector size for each sentence we want
# max_features: Total number of different feature vectors used such as pos tag, top2vec, doc2vec, tfidf
# num_class: Number of output class such as sad, happy, angry
def create_model(feature_size, max_features, num_class):
    feature_input = Input((max_features,feature_size), dtype=tf.float32)
    
    feature_vectors = TimeDistributed(Dense(feature_size, use_bias=False, activation='tanh'))(feature_input)
    
    # Attention Layer
    attention_vectors = Dense(1)(feature_vectors)
    attention_weights = Softmax(axis=1)(attention_vectors)
    
    #attention_weights = Dense(max_features, use_bias=False, activation='softmax')(feature_vectors)
    
    # Generating code vectors
    text_vectors = K.sum(feature_vectors * attention_weights, axis=1)
    
    # Prediction layer
    output_class = Dense(num_class, use_bias=False, activation='softmax')(text_vectors)
    
    model = Model(inputs=feature_input, outputs=output_class)
    return model

from sklearn.metrics import classification_report

feature_size = 1
max_features = 10
num_class = 2
xtrain, xval, ytrain, yval = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=4)
xtrain = xtrain.astype('float32')
ytrain = ytrain.astype('int64')  # or 'float32' if one-hot encoded
xval = xval.astype('float32')
yval = yval.astype('int64')  # or 'float32' if one-hot encoded

from tensorflow.keras.utils import to_categorical

print(ytrain.shape)

ytrain = to_categorical(ytrain)  # Replace num_classes with the number of classes
yval = to_categorical(yval)

model = create_model(feature_size, max_features, num_class)
#compile model
model.compile(loss='categorical_crossentropy',
                       optimizer='adam',
                       metrics=['acc'])
# check summary of model
model.summary()

# Early stopping
earlystopping = callbacks.EarlyStopping(monitor ="val_loss", 
                                    mode ="min", patience = 5, 
                                    restore_best_weights = True)

# train model
# train model
history = model.fit(x=xtrain, y=ytrain, batch_size=32, epochs=20, 
                    validation_data=(xval, yval), callbacks=[earlystopping])
# model.fit(x=xtrain, y=ytrain,batch_size=32,epochs=20, 
#           validation_data=(xval, yval), callbacks =[earlystopping])

for layer in model.layers:
    print(layer.name)
    
# Performance
predicted = model.predict(x=xval)
predicted = np.where(predicted > 0.5, 1, 0)
print(classification_report(yval, predicted, digits=2))

# Extract loss values from the history object
train_loss = history.history['loss']
train_loss = [round(num, 2) for num in train_loss]
val_loss = history.history['val_loss']
val_loss = [round(num, 2) for num in val_loss]
epochs = [i + 1 for i in range(len(train_loss))]
print(train_loss)
print(val_loss)
print(epochs)

# Plotting the training and validation loss
# plt.figure(figsize=(8, 6))
# plt.plot(epochs, train_loss, 'b', label='Training Loss')
# plt.plot(epochs, val_loss, 'r', label='Validation Loss')
# plt.title('Training and Validation Loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()

import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Conv1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix, accuracy_score

#A very simple ANN model with 2 layers 
model = Sequential([
        Dense(X_train.shape[1], activation="relu"),
        Dense(X_train.shape[1]/2, activation="relu"),
        Dense(1, activation = 'sigmoid'),
    ])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics = ['accuracy'])


xtrain, xval, ytrain, yval = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=4)
xtrain = xtrain.astype('float32')
ytrain = ytrain.astype('int64')  # or 'float32' if one-hot encoded
xval = xval.astype('float32')
yval = yval.astype('int64')  # or 'float32' if one-hot encoded

history = model.fit(x=xtrain, 
          y=ytrain, 
          epochs=20,
          validation_data=(xval, yval), verbose=0
          )

predictions = np.round(model.predict(xval))
print(classification_report(yval,predictions))

# Extract loss values from the history object
train_loss = history.history['loss']
train_loss = [round(num, 2) for num in train_loss]
val_loss = history.history['val_loss']
val_loss = [round(num, 2) for num in val_loss]
epochs = [i + 1 for i in range(len(train_loss))]
print(train_loss)
print(val_loss)
print(epochs)

