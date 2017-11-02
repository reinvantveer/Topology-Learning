import os
from datetime import datetime

import numpy as np
from keras import Input
from keras.callbacks import TensorBoard, EarlyStopping
from keras.engine import Model
from keras.layers import LSTM, Dense, LeakyReLU, TimeDistributed, Flatten
from keras.optimizers import Adam
from matplotlib import pyplot as plt
from topoml_util.ConsoleLogger import DecypherAll
from topoml_util.gaussian_loss import univariate_gaussian_loss
from topoml_util.geom_scaler import localized_normal, localized_mean
from topoml_util.slack_send import notify

SCRIPT_VERSION = "0.0.8"
SCRIPT_NAME = os.path.basename(__file__)
TIMESTAMP = str(datetime.now()).replace(':', '.')
SIGNATURE = SCRIPT_NAME + ' ' + TIMESTAMP
PLOT_DIR = 'plots/' + SIGNATURE
DATA_FILE = '../files/geodata_vectorized.npz'
BATCH_SIZE = 512
TRAIN_VALIDATE_SPLIT = 0.1
LATENT_SIZE = 128
EPOCHS = 400
OPTIMIZER = Adam(lr=1e-3)

loaded = np.load(DATA_FILE)
training_vectors = loaded['input_geoms']

# Bring coordinates and distance in the same scale
means = localized_mean(training_vectors)
training_vectors = localized_normal(training_vectors, means, 1e4)

(data_points, max_points, GEO_VECTOR_LEN) = training_vectors.shape
target_vectors = loaded['geom_distance'][:, 0, :]

inputs = Input(name='Input', shape=(max_points, GEO_VECTOR_LEN))
model = LSTM(LATENT_SIZE, activation='relu', return_sequences=True)(inputs)
model = TimeDistributed(Dense(32))(model)
model = LeakyReLU()(model)
model = LSTM(LATENT_SIZE, activation='relu', return_sequences=True)(model)
model = TimeDistributed(Dense(32))(model)
model = Flatten()(model)
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

val_set_start = -round(data_points * TRAIN_VALIDATE_SPLIT)
prediction = model.predict(training_vectors[val_set_start:])
error = prediction[:, 0] - target_vectors[val_set_start:, 0]
fig, ax = plt.subplots()
plt.text(0.01, 0.94, r'prediction error $\mu: $' + str(np.round(np.mean(error), 4)), transform=ax.transAxes)
plt.text(0.01, 0.88, r'prediction error $\sigma: $' + str(np.round(np.std(error), 4)), transform=ax.transAxes)
plt.xlabel('Error')
plt.ylabel('Frequency')
plt.title('Histogram error frequency')
n, bins, patches = plt.hist(error, 50, facecolor='g', normed=False, alpha=0.75)
os.makedirs(str(PLOT_DIR), exist_ok=True)
plt.savefig(PLOT_DIR + '/plt_' + TIMESTAMP + '.png')

notify(TIMESTAMP, SCRIPT_NAME, 'validation loss of ' + str(history['val_loss'][-1]))
print(SCRIPT_NAME, 'finished successfully')
