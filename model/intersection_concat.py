import os
from datetime import datetime
from shutil import copyfile

import numpy as np
from keras import Input
from keras.callbacks import TensorBoard, EarlyStopping
from keras.engine import Model
from keras.layers import LSTM, Dense, concatenate, TimeDistributed, RepeatVector
from keras.optimizers import Adam
from topoml_util.GaussianMixtureLoss import GaussianMixtureLoss
from topoml_util.GeoVectorizer import ONE_HOT_LEN
from topoml_util.PyplotLogger import DecypherAll
from topoml_util.geom_scaler import localized_normal, localized_mean
from topoml_util.slack_send import notify

SCRIPT_VERSION = "0.0.8"
SCRIPT_NAME = os.path.basename(__file__)
TIMESTAMP = str(datetime.now()).replace(':', '.')
SIGNATURE = SCRIPT_NAME + ' ' + TIMESTAMP
PLOT_DIR = './plots/' + TIMESTAMP + ' ' + SCRIPT_NAME
DATA_FILE = '../files/geodata_vectorized.npz'
BATCH_SIZE = 1024
GAUSSIAN_MIXTURE_COMPONENTS = 1
TRAIN_VALIDATE_SPLIT = 0.1
LSTM_SIZE = 128
DENSE_SIZE = 32
REPEAT_HIDDEN = 2
EPOCHS = 400
OPTIMIZER = Adam(lr=1e-3)

# Archive the configuration
copyfile(__file__, 'configs/' + SIGNATURE)

loaded = np.load(DATA_FILE)
raw_brt_vectors = loaded['brt_vectors']
raw_osm_vectors = loaded['osm_vectors']
raw_target_vectors = loaded['intersection']

brt_vectors = []
osm_vectors = []
target_vectors = []

# skip non-intersecting geometries
for brt, osm, target in zip(raw_brt_vectors, raw_osm_vectors, raw_target_vectors):
    if not target[0, 0] == 0:  # a zero coordinate designates an empty geometry
        brt_vectors.append(brt)
        osm_vectors.append(osm)
        target_vectors.append(target)

# data whitening
means = localized_mean(target_vectors)
brt_vectors = localized_normal(brt_vectors, means, 1e4)
osm_vectors = localized_normal(osm_vectors, means, 1e4)
target_vectors = localized_normal(target_vectors, means, 1e4)

# shape determination
(data_points, brt_max_points, BRT_INPUT_VECTOR_LEN) = brt_vectors.shape
(_, osm_max_points, OSM_INPUT_VECTOR_LEN) = osm_vectors.shape
target_max_points = target_vectors.shape[1]
output_seq_length = (GAUSSIAN_MIXTURE_COMPONENTS * 6) + ONE_HOT_LEN
output_size_2d = target_max_points * output_seq_length

Loss = GaussianMixtureLoss(num_components=GAUSSIAN_MIXTURE_COMPONENTS, num_points=target_max_points)

brt_inputs = Input(shape=(brt_max_points, BRT_INPUT_VECTOR_LEN))
brt_model = LSTM(brt_max_points * 2, activation='relu')(brt_inputs)

osm_inputs = Input(shape=(osm_max_points, OSM_INPUT_VECTOR_LEN))
osm_model = LSTM(osm_max_points * 2, activation='relu')(osm_inputs)

concat = concatenate([brt_model, osm_model])
model = RepeatVector(target_max_points)(concat)

for layer in range(REPEAT_HIDDEN):
    model = LSTM(LSTM_SIZE, activation='relu', return_sequences=True)(model)

model = TimeDistributed(Dense(DENSE_SIZE, activation='relu'))(model)
model = Dense(output_seq_length)(model)

model = Model(inputs=[brt_inputs, osm_inputs], outputs=model)
model.compile(
    loss=Loss.geom_gaussian_mixture_loss,
    optimizer=OPTIMIZER)
model.summary()

# Callbacks
callbacks = [
    TensorBoard(log_dir='./tensorboard_log/' + TIMESTAMP + ' ' + SCRIPT_NAME, write_graph=False),
    DecypherAll(gmm_size=GAUSSIAN_MIXTURE_COMPONENTS,
                plot_dir=PLOT_DIR,
                input_slice=lambda x: x[:2],
                target_slice=lambda x: x[2:3],
                stdout=False),
    EarlyStopping(patience=40, min_delta=1)
]

history = model.fit(
    x=[brt_vectors, osm_vectors],
    y=target_vectors,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=TRAIN_VALIDATE_SPLIT,
    callbacks=callbacks).history

notify(TIMESTAMP, SCRIPT_NAME, 'validation loss of ' + str(history['val_loss'][-1]))
print(SCRIPT_NAME, 'finished successfully')
