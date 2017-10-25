import os
from datetime import datetime
from shutil import copyfile

import numpy as np
from keras import Input
from keras.callbacks import TensorBoard
from keras.engine import Model
from keras.layers import LSTM, Dense
from keras.optimizers import Adam
from topoml_util.slack_send import notify
from topoml_util.GaussianMixtureLoss import GaussianMixtureLoss
from topoml_util.GeoVectorizer import ONE_HOT_LEN
from topoml_util.PyplotLogger import DecypherAll
from topoml_util.geom_scaler import localized_normal, localized_mean

# To suppress tensorflow info level messages:
# export TF_CPP_MIN_LOG_LEVEL=2

SCRIPT_VERSION = "0.0.2"
SCRIPT_NAME = os.path.basename(__file__)
TIMESTAMP = str(datetime.now()).replace(':', '.')
PLOT_DIR = './plots/' + TIMESTAMP + ' ' + SCRIPT_NAME
DATA_FILE = '../files/geodata_vectorized.npz'
BATCH_SIZE = 1024
GAUSSIAN_MIXTURE_COMPONENTS = 20
TRAIN_VALIDATE_SPLIT = 0.1
REPEAT_DEEP_ARCH = 0
LSTM_SIZE = 128
DENSE_SIZE = 64
EPOCHS = 400
OPTIMIZER = Adam(lr=1e-4, clipnorm=1.)

# Archive the configuration
copyfile(__file__, 'configs/' + TIMESTAMP + ' ' + SCRIPT_NAME)

loaded = np.load(DATA_FILE)
raw_training_vectors = loaded['input_geoms']
raw_target_vectors = loaded['intersection']

training_vectors = []
target_vectors = []

# skip non-intersecting geometries
for train, target in zip(raw_training_vectors, raw_target_vectors):
    if not target[0, 0] == 0:  # a zero coordinate designates an empty geometry
        training_vectors.append(train)
        target_vectors.append(target)

means = localized_mean(training_vectors)
training_vectors = localized_normal(training_vectors, means, 1e4)
target_vectors = localized_normal(target_vectors, means, 1e4)

(data_points, max_points, INPUT_VECTOR_LEN) = training_vectors.shape
# Expand the target vectors to gaussian mixture model size compatible with the prediction format
component_1 = target_vectors[:, :, 0:5]  # The first gaussian mixture model component for each point
component_1 = np.append(component_1, np.zeros((data_points, max_points, 1)), axis=2)  # Add pi feature
target_vectors = np.append(
    np.reshape(
        np.repeat(component_1, GAUSSIAN_MIXTURE_COMPONENTS, axis=1),
        (data_points, max_points, 6 * GAUSSIAN_MIXTURE_COMPONENTS)),
    target_vectors[:, :, 5:], axis=2)

output_size = GAUSSIAN_MIXTURE_COMPONENTS * 6 + ONE_HOT_LEN
Loss = GaussianMixtureLoss(GAUSSIAN_MIXTURE_COMPONENTS, max_points)

inputs = Input(shape=(max_points, INPUT_VECTOR_LEN))
model = LSTM(LSTM_SIZE, activation='relu', return_sequences=True)(inputs)

for layer in range(REPEAT_DEEP_ARCH):
    model = LSTM(LSTM_SIZE, activation='relu', return_sequences=True)(model)
    model = Dense(DENSE_SIZE, activation='relu')(model)

model = Dense(output_size)(model)
model = Model(inputs, model)
model.compile(
    loss=Loss.geom_gaussian_mixture_loss,
    optimizer=OPTIMIZER)
model.summary()

tb_callback = TensorBoard(log_dir='./tensorboard_log/' + TIMESTAMP + ' ' + SCRIPT_NAME, write_graph=False)
decypher = DecypherAll(gmm_size=GAUSSIAN_MIXTURE_COMPONENTS, plot_dir=PLOT_DIR)

history = model.fit(
    x=training_vectors,
    y=target_vectors,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=TRAIN_VALIDATE_SPLIT,
    callbacks=[decypher, tb_callback]).history

notify(TIMESTAMP, SCRIPT_NAME, 'validation loss of ' + str(history['val_loss'][-1]))
print(SCRIPT_NAME, 'finished successfully')
