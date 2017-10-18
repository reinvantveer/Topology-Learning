import os
from datetime import datetime
from shutil import copyfile

import numpy as np
from keras import Input
from keras.callbacks import TensorBoard, EarlyStopping
from keras.engine import Model
from keras.layers import LSTM, Dense, Reshape
from keras.optimizers import Adam
from slackclient import SlackClient

from topoml_util.GaussianMixtureLoss import GaussianMixtureLoss
from topoml_util.GeoVectorizer import GEOM_TYPE_LEN, RENDER_LEN
from topoml_util.PyplotLogger import DecypherAll

# To suppress tensorflow info level messages:
# export TF_CPP_MIN_LOG_LEVEL=2

SCRIPT_NAME = os.path.basename(__file__)
TIMESTAMP = str(datetime.now()).replace(':', '.')
PLOT_DIR = './plots/' + TIMESTAMP + ' ' + SCRIPT_NAME
DATA_FILE = '../files/densified_vectorized.npz'
BATCH_SIZE = 1024
GAUSSIAN_MIXTURE_COMPONENTS = 1
DENSIFIED = 100
TRAIN_VALIDATE_SPLIT = 0.1
REPEAT_DEEP_ARCH = 2
LSTM_SIZE = 128
DENSE_SIZE = 64
EPOCHS = 400
OPTIMIZER = Adam(lr=1e-3, clipnorm=1.)

# Archive the configuration
copyfile(__file__, 'configs/' + TIMESTAMP + ' ' + SCRIPT_NAME)

loaded = np.load(DATA_FILE)
input_vectors = loaded['input_geoms']
target_vectors = loaded['intersection']

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

model.fit(
    x=input_vectors,
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

