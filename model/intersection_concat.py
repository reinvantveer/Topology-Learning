import os
from datetime import datetime
from shutil import copyfile

import numpy as np
from keras import Input
from keras.callbacks import TensorBoard, EarlyStopping
from keras.engine import Model
from keras.layers import LSTM, Dense, concatenate, Reshape, RepeatVector, TimeDistributed, LeakyReLU
from keras.optimizers import Adam
from slackclient import SlackClient

from topoml_util.GaussianMixtureLoss import GaussianMixtureLoss
from topoml_util.GeoVectorizer import ONE_HOT_LEN
from topoml_util.PyplotLogger import DecypherAll
from topoml_util.geom_scaler import localized_normal, localized_mean

# To suppress tensorflow info level messages:
# export TF_CPP_MIN_LOG_LEVEL=2

SCRIPT_NAME = os.path.basename(__file__)
TIMESTAMP = str(datetime.now()).replace(':', '.')
PLOT_DIR = './plots/' + TIMESTAMP + ' ' + SCRIPT_NAME
DATA_FILE = '../files/geodata_vectorized.npz'
BATCH_SIZE = 1024
GAUSSIAN_MIXTURE_COMPONENTS = 1
TRAIN_VALIDATE_SPLIT = 0.1
REPEAT_DEEP_ARCH = 2
LSTM_SIZE = 150
DENSE_SIZE = 64
EPOCHS = 400
OPTIMIZER = Adam(lr=1e-3, clipnorm=1.)

# Archive the configuration
copyfile(__file__, 'configs/' + TIMESTAMP + ' ' + SCRIPT_NAME)

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

means = localized_mean(target_vectors)
brt_vectors = localized_normal(brt_vectors, means, 1e4)
osm_vectors = localized_normal(osm_vectors, means, 1e4)
target_vectors = localized_normal(target_vectors, means, 1e4)

(data_points, brt_max_points, BRT_INPUT_VECTOR_LEN) = brt_vectors.shape
(_, osm_max_points, OSM_INPUT_VECTOR_LEN) = osm_vectors.shape
target_max_points = target_vectors.shape[1]
output_seq_length = (GAUSSIAN_MIXTURE_COMPONENTS * 6) + ONE_HOT_LEN
output_size = target_max_points * output_seq_length

Loss = GaussianMixtureLoss(num_components=GAUSSIAN_MIXTURE_COMPONENTS, num_points=target_max_points)

brt_inputs = Input(shape=(brt_max_points, BRT_INPUT_VECTOR_LEN))
brt_model = LSTM(brt_max_points * 2, activation='relu')(brt_inputs)

osm_inputs = Input(shape=(osm_max_points, OSM_INPUT_VECTOR_LEN))
osm_model = LSTM(osm_max_points * 2, activation='relu')(osm_inputs)

concat = concatenate([brt_model, osm_model])
model = RepeatVector(target_max_points)(concat)
model = LSTM(LSTM_SIZE, activation='relu', return_sequences=True)(model)
model = LSTM(LSTM_SIZE, activation='relu', return_sequences=True)(model)
model = LSTM(LSTM_SIZE, activation='relu', return_sequences=True)(model)
model = TimeDistributed(Dense(256, activation='relu'))(model)
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
                target_slice=lambda x: x[2:3]),
    EarlyStopping(patience=40)
]

model.fit(
    x=[brt_vectors, osm_vectors],
    y=target_vectors,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=TRAIN_VALIDATE_SPLIT,
    callbacks=callbacks)

slack_token = os.environ.get("SLACK_API_TOKEN")

if slack_token:
    sc = SlackClient(slack_token)
    sc.api_call(
        "chat.postMessage",
        channel="#machinelearning",
        text="Session " + TIMESTAMP + ' ' + SCRIPT_NAME + " completed successfully")
else:
    print('No slack notification: no slack API token environment variable "SLACK_API_TOKEN" set.')

print('Done!')
