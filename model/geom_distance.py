import os
from datetime import datetime

import numpy as np
from keras import Input
from keras.callbacks import TensorBoard, EarlyStopping
from keras.engine import Model
from keras.layers import LSTM, Dense, TimeDistributed
from keras.optimizers import Adam
from topoml_util.ConsoleLogger import DecypherAll
from topoml_util.gaussian_loss import univariate_gaussian_loss
from topoml_util.geom_scaler import localized_normal, localized_mean
from topoml_util.slack_send import notify

SCRIPT_VERSION = "0.0.4"
SCRIPT_NAME = os.path.basename(__file__)
TIMESTAMP = str(datetime.now()).replace(':', '.')
SIGNATURE = SCRIPT_NAME + ' ' + TIMESTAMP
DATA_FILE = '../files/geodata_vectorized.npz'
BATCH_SIZE = 512
TRAIN_VALIDATE_SPLIT = 0.1
LATENT_SIZE = 128
EPOCHS = 50
OPTIMIZER = Adam(lr=0.005)

loaded = np.load(DATA_FILE)
training_vectors = loaded['input_geoms']

# Bring coordinates and distance in the same scale
means = localized_mean(training_vectors)
training_vectors = localized_normal(training_vectors, means, 1e4)

(data_points, max_points, GEO_VECTOR_LEN) = training_vectors.shape
target_vectors = loaded['geom_distance'][:, 0, :]

inputs = Input(name='Input', shape=(max_points, GEO_VECTOR_LEN))
model = LSTM(LATENT_SIZE, activation='relu')(inputs)
model = TimeDistributed(Dense(32))(model)
model = Dense(2)(model)
model = Model(inputs, model)
model.compile(
    loss=univariate_gaussian_loss,
    optimizer=OPTIMIZER)
model.summary()

callbacks = [
    TensorBoard(log_dir='./tensorboard_log/' + TIMESTAMP + ' ' + SCRIPT_NAME, write_graph=False),
    DecypherAll(lambda x: str(x)),
    EarlyStopping(patience=40, min_delta=0.001)
]


history = model.fit(x=training_vectors,
                    y=target_vectors,
                    epochs=EPOCHS,
                    batch_size=BATCH_SIZE,
                    validation_split=TRAIN_VALIDATE_SPLIT,
                    callbacks=callbacks).history

prediction = model.predict(training_vectors[0:1000])

intersecting_target = []
intersecting_prediction = []
non_intersecting_target = []
non_intersecting_prediction = []

for index, _ in enumerate(target_vectors[0:1000]):
    if target_vectors[index, 0] == 0.:
        intersecting_target.append(target_vectors[index])
        intersecting_prediction.append(prediction[index])
    else:
        non_intersecting_target.append(target_vectors[index])
        non_intersecting_prediction.append(prediction[index])

intersecting_error = np.abs(np.array(intersecting_prediction)[:, 0] - np.array(intersecting_target)[:, 0])
non_intersecting_error = np.abs(np.array(non_intersecting_prediction)[:, 0] - np.array(non_intersecting_target)[:, 0])
print('Intersecting error factor:', np.mean(intersecting_error))
print('Non-intersecting error factor:', np.mean(non_intersecting_error))

notify(TIMESTAMP, SCRIPT_NAME, 'validation loss of ' + str(history['val_loss'][-1]))
print(SCRIPT_NAME, 'finished successfully')
