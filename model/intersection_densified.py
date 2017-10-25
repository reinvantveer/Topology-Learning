import os
from datetime import datetime

import numpy as np
from keras import Input
from keras.callbacks import TensorBoard, EarlyStopping
from keras.engine import Model
from keras.layers import LSTM, Dense, Reshape
from keras.optimizers import Adam
from topoml_util.geom_scaler import localized_mean, localized_normal
from topoml_util.GaussianMixtureLoss import GaussianMixtureLoss
from topoml_util.GeoVectorizer import GEOM_TYPE_LEN, RENDER_LEN, GeoVectorizer
from topoml_util.PyplotLogger import DecypherAll
from topoml_util.slack_send import notify

SCRIPT_VERSION = "0.0.2"
SCRIPT_NAME = os.path.basename(__file__)
TIMESTAMP = str(datetime.now()).replace(':', '.')
PLOT_DIR = './plots/' + TIMESTAMP + ' ' + SCRIPT_NAME
DATA_FILE = '../files/geodata_vectorized.npz'
BATCH_SIZE = 1024
GAUSSIAN_MIXTURE_COMPONENTS = 1
DENSIFIED = 20
TRAIN_VALIDATE_SPLIT = 0.1
REPEAT_DEEP_ARCH = 2
LSTM_SIZE = 128
DENSE_SIZE = 64
EPOCHS = 400
OPTIMIZER = Adam(lr=1e-3, clipnorm=1.)

TARGET_FILE = '../files/densified_vectorized.npz'

loaded = np.load(DATA_FILE)
raw_training_vectors = loaded['input_geoms']
raw_target_vectors = loaded['intersection']

input_vectors = []
target_vectors = []

# skip non-intersecting geometries
for train, target in zip(raw_training_vectors, raw_target_vectors):
    if not target[0, 0] == 0:  # a zero coordinate designates an empty geometry
        input_vectors.append(train)
        target_vectors.append(target)

print('Preprocessing vectors...')
means = localized_mean(input_vectors)
input_vectors = localized_normal(input_vectors, means, 1e4)
input_vectors = np.array([GeoVectorizer.interpolate(vector, DENSIFIED) for vector in input_vectors])
target_vectors = localized_normal(target_vectors, means, 1e4)
target_vectors = np.array([GeoVectorizer.interpolate(vector, 50) for vector in target_vectors])

(data_points, max_input_points, INPUT_VECTOR_LEN) = input_vectors.shape
(_, max_target_points, _) = target_vectors.shape

output_size = GAUSSIAN_MIXTURE_COMPONENTS * 6 + GEOM_TYPE_LEN + RENDER_LEN
Loss = GaussianMixtureLoss(GAUSSIAN_MIXTURE_COMPONENTS, max_target_points)

inputs = Input(shape=(max_input_points, INPUT_VECTOR_LEN))
model = Dense(INPUT_VECTOR_LEN, activation='relu')(inputs)

for layer in range(REPEAT_DEEP_ARCH):
    model = LSTM(LSTM_SIZE, activation='relu', return_sequences=True)(model)
    model = Dense(DENSE_SIZE, activation='relu')(model)

model = Reshape((max_target_points, DENSE_SIZE * 2))(model)
model = Dense(output_size)(model)
model = Model(inputs, model)
model.compile(
    loss=Loss.geom_gaussian_mixture_loss,
    optimizer=OPTIMIZER)
model.summary()

callbacks = [
    TensorBoard(log_dir='./tensorboard_log/' + TIMESTAMP + ' ' + SCRIPT_NAME, write_graph=False),
    DecypherAll(gmm_size=GAUSSIAN_MIXTURE_COMPONENTS, plot_dir=PLOT_DIR),
    EarlyStopping(patience=40, min_delta=1e-3)
]

history = model.fit(
    x=input_vectors,
    y=target_vectors,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=TRAIN_VALIDATE_SPLIT,
    callbacks=callbacks).history

notify(TIMESTAMP, SCRIPT_NAME, 'validation loss of ' + str(history['val_loss'][-1]))
print(SCRIPT_NAME, 'finished successfully')
