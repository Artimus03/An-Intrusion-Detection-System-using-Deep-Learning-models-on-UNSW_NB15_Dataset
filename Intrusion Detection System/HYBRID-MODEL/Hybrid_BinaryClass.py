#HYBRID CNN+LSTM

#IMPORTING LIBRARIES
 import numpy as np
 import pandas as pd
 from sklearn.model_selection import train_test_split
 from keras.models import Sequential
 from keras.layers import Conv1D, MaxPooling1D, Dropout, Dense, LSTM
 from keras.optimizers import Adam
 from keras.callbacks import EarlyStopping, ModelCheckpoint
 from plot import plot_training_vs_validation
 from sklearn.metrics import confusion_matrix, classification_report

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
 
 hybrid_model_bin = Sequential()
 hybrid_model_bin.add(Conv1D(
	filters=64,
	kernel_size=3,
	activation='relu',
	input_shape=(X_train_reshaped.shape[1], 1)
	)
 )
 
 hybrid_model_bin.add(MaxPooling1D(pool_size=2))
 hybrid_model_bin.add(LSTM(units=100, activation='relu', return_sequences=True))
 hybrid_model_bin.add(Dropout(0.35))
 hybrid_model_bin.add(LSTM(units=60, activation='relu'))
 hybrid_model_bin.add(Dropout(0.15))
 hybrid_model_bin.add(Dense(units=10, activation='softmax'))
 
 adam_optimizer = Adam(learning_rate=1e-4)
	 hybrid_model_bin.compile(
	 loss='sparse_categorical_crossentropy',
	 optimizer=adam_optimizer,
	 metrics=['accuracy']
 )
 callbacks = [
	 EarlyStopping(monitor='val_loss', patience=10),
	 ModelCheckpoint(
	 filepath='best_model_cnn.h5',
	 monitor='val_loss',
	 save_best_only=True
	)
 ]
 hybrid_model_bin_history = hybrid_model_bin.fit(
	 X_train_reshaped, y_train,
	 epochs=50,
	 batch_size=256,
	 validation_data=(X_test_reshaped, y_test),
	 callbacks=callbacks
 )
 
 plot_training_vs_validation(hybrid_model_bin_history, "CNN-LSTM | Binary")
 
 y_test_probabilities = hybrid_model_bin_history.predict(X_test_reshaped)
 y_test_pred = np.argmax(y_test_probabilities, axis=1)
 
 conf_matrix = confusion_matrix(y_test, y_test_pred)
 print("Confusion Matrix:")
 print(conf_matrix)
 
 print("\nClassification Report:")
 print(classification_report(y_test, y_test_pred))