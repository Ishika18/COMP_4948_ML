from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint

# configure early stopping
es = EarlyStopping(monitor='val_loss', patience=10)
mc = ModelCheckpoint('best_model.h5', monitor='val_accuracy', mode='max', verbose=1,
                     save_best_only=True)

history = model.fit(X_train, y_train, epochs=20, batch_size=100,
                    validation_data=(X_test, y_test), callbacks=[es,mc])

""""
replace
predictions = model.predict(X_test)

with

# Must match from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models    import load_model
saved_model = load_model('best_model.h5')
predictions = saved_model.predict(X_test)

"""