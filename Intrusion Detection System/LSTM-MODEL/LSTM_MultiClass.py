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
 
#MULTI CLASS CLASSIFICATION
 X=multi_data.drop(columns=['label'],axis=1)
 Y=multi_data['label']
 X_train, X_test, y_train, y_test = train_test_split(
	X, Y, test_size=0.20, random_state=50
 )
 X_train_reshaped = X_train.values.reshape((X_train.shape[0], X_train.shape[1], 1))
 X_test_reshaped = X_test.values.reshape((X_test.shape[0], X_test.shape[1], 1))
 lstm_model_multi = Sequential()
 lstm_model_multi.add(LSTM(
	64, activation='relu', input_shape=(69, 1))
 )
 
 lstm_model_multi.add(Dense(64, activation='relu'))
 lstm_model_multi.add(Dense(num_classes, activation='softmax'))
 
 adam_optimizer = Adam(learning_rate=1e-4)
	lstm_model_multi.compile(
	loss='sparse_categorical_crossentropy',
	optimizer=adam_optimizer,
	metrics=['accuracy']
 )
 
 callbacks = [
	EarlyStopping(monitor='val_loss', patience=10),
	ModelCheckpoint(
	filepath='best_model_cnn.h5',
	monitor='val_loss', save_best_only=True
	)
 ]
 
 lstm_history_multi = lstm_model_multi.fit(
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