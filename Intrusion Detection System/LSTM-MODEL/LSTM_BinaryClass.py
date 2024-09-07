# LONG SHORT-TERM MEMORY

#IMPORTING LIBRARIES
 import numpy as np
 import pandas as pd
 from sklearn.model_selection import train_test_split
 from keras.models import Sequential
 from keras.layers import Dense, LSTM
 from plot import plot_training_vs_validation
 from sklearn.metrics import confusion_matrix, classification_report
 from keras.optimizers import Adam
 from keras.callbacks import EarlyStopping, ModelCheckpoint
 
#IMPORTING DATASETS
 bin_data = pd.read_csv("bin_data.csv")
 multi_data = pd.read_csv("multi_data.csv")
 
#BINARY CLASS CLASSIFICATION
 X=bin_data.drop(columns=['label'],axis=1)
 Y=bin_data['label']
 X_train, X_test, y_train, y_test = train_test_split(
	X, Y, test_size=0.20, random_state=50
 )
 X_train_reshaped = X_train.values.reshape((X_train.shape[0], X_train.shape[1], 1))
 X_test_reshaped = X_test.values.reshape((X_test.shape[0], X_test.shape[1], 1))
 
 unique_labels = np.unique(y_train)
 num_classes = len(unique_labels)
 print("Unique Labels:", unique_labels)
 print("Number of Classes:", num_classes)
 
 lstm_model = Sequential()
 lstm_model.add(LSTM(64, activation='relu', input_shape=(14, 1)))
 lstm_model.add(Dense(64, activation='relu'))
 lstm_model.add(Dense(num_classes, activation='softmax'))
 
 adam_optimizer = Adam(learning_rate=1e-4)
 lstm_model.compile(
	loss='sparse_categorical_crossentropy',
	optimizer=adam_optimizer,
	metrics=['accuracy']
 )
 
 lstm_history = lstm_model.fit(
	X_train_reshaped, y_train,
	epochs=50,
	batch_size=256,
	validation_data=(X_test_reshaped, y_test),
	callbacks=callbacks
 )
 
 plot_training_vs_validation(lstm_history, "LSTM")
 y_test_probabilities = lstm_model.predict(X_test_reshaped)
 y_test_pred = np.argmax(y_test_probabilities, axis=1)
 
 conf_matrix = confusion_matrix(y_test, y_test_pred)
 print("Confusion Matrix:")
 print(conf_matrix)
 
 print("\nClassification Report:")
 print(classification_report(y_test, y_test_pred))