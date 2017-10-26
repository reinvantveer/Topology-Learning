import os
from datetime import datetime

import numpy as np
from keras import Input
from keras.callbacks import TensorBoard, EarlyStopping
from keras.engine import Model
from keras.layers import LSTM, Dense, concatenate, Reshape
from keras.optimizers import Adam
from topoml_util.LoggerCallback import EpochLogger
from topoml_util.gaussian_loss import univariate_gaussian_loss
from topoml_util.geom_scaler import localized_normal, localized_mean
from topoml_util.slack_send import notify

SCRIPT_VERSION = "0.0.7"
SCRIPT_NAME = os.path.basename(__file__)
TIMESTAMP = str(datetime.now()).replace(':', '.')
SIGNATURE = SCRIPT_NAME + ' ' + TIMESTAMP
DATA_FILE = '../files/geodata_vectorized.npz'
BATCH_SIZE = 1024
TRAIN_VALIDATE_SPLIT = 0.1
HIDDEN_SIZE = 128
REPEAT_HIDDEN = 3
EPOCHS = 400
OPTIMIZER = Adam(lr=1e-3)

loaded = np.load(DATA_FILE)
raw_brt_vectors = loaded['brt_vectors']
raw_osm_vectors = loaded['osm_vectors']
raw_intersection_vectors = loaded['intersection']

brt_vectors = []
osm_vectors = []
intersection_vectors = []

# skip non-intersecting geometries
for brt, osm, target in zip(raw_brt_vectors, raw_osm_vectors, raw_intersection_vectors):
    if not target[0, 0] == 0:  # a zero coordinate designates an empty geometry
        brt_vectors.append(brt)
        osm_vectors.append(osm)
        intersection_vectors.append(target)

# data whitening
means = localized_mean(intersection_vectors)
brt_vectors = localized_normal(brt_vectors, means, 1e4)
osm_vectors = localized_normal(osm_vectors, means, 1e4)
intersection_vectors = localized_normal(intersection_vectors, means, 1e4)

# shape determination
(data_points, brt_max_points, brt_seq_len) = brt_vectors.shape
(_, osm_max_points, osm_seq_len) = osm_vectors.shape


brt_inputs = Input(shape=(brt_max_points, brt_seq_len))
brt_model = LSTM(brt_max_points * 2, activation='relu')(brt_inputs)

osm_inputs = Input(shape=(osm_max_points, osm_seq_len))
osm_model = LSTM(osm_max_points * 2, activation='relu')(osm_inputs)

concat = concatenate([brt_model, osm_model])
model = Reshape((1, brt_seq_len + osm_seq_len))(concat)

for layer in range(REPEAT_HIDDEN):
    model = LSTM(HIDDEN_SIZE, activation='relu', return_sequences=True)(model)
    model = Dense(32, activation='relu')(model)

model = LSTM(HIDDEN_SIZE, activation='relu')(model)  # Flatten
model = Dense(2)(model)
model = Model(inputs=[brt_inputs, osm_inputs], outputs=model)
model.compile(loss=univariate_gaussian_loss, optimizer=OPTIMIZER)
model.summary()

callbacks = [
    TensorBoard(log_dir='./tensorboard_log/' + SIGNATURE, write_graph=False),
    EpochLogger(),
    EarlyStopping(patience=40, min_delta=0.001)
]

history = model.fit(
    x=[brt_vectors, osm_vectors],
    y=intersection_vectors,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=TRAIN_VALIDATE_SPLIT,
    callbacks=callbacks).history

notify(TIMESTAMP, SCRIPT_NAME, 'validation loss of ' + str(history['val_loss'][-1]))
print(SCRIPT_NAME, 'finished successfully')
