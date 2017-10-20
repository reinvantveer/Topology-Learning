from datetime import datetime
import os
import numpy as np

from keras import Input
from keras.callbacks import TensorBoard, EarlyStopping
from keras.engine import Model
from keras.layers import LSTM, Dense
from keras.optimizers import Adam

from topoml_util.geom_scaler import localized_normal, localized_mean
from topoml_util.ConsoleLogger import DecypherAll
from topoml_util.gaussian_loss import univariate_gaussian_loss

# To suppress tensorflow info level messages:
# export TF_CPP_MIN_LOG_LEVEL=2
from topoml_util.slack_send import notify

SCRIPT_VERSION = "0.0.1"
TIMESTAMP = str(datetime.now()).replace(':', '.')
SCRIPT_NAME = os.path.basename(__file__)
DATA_FILE = '../files/geodata_vectorized.npz'
BATCH_SIZE = 512
TRAIN_VALIDATE_SPLIT = 0.1
LATENT_SIZE = 64
EPOCHS = 50
OPTIMIZER = Adam(lr=0.001)

loaded = np.load(DATA_FILE)
training_vectors = loaded['input_geoms']
(data_points, max_points, GEO_VECTOR_LEN) = training_vectors.shape

# Bring coordinates and distance in roughly the same scale
means = localized_mean(training_vectors)
training_vectors = localized_normal(training_vectors, means, 1e4)
target_vectors = loaded['centroid_distance'][:, 0, :]

inputs = Input(name='Input', shape=(max_points, GEO_VECTOR_LEN))
model = LSTM(LATENT_SIZE, activation='relu')(inputs)
model = Dense(2)(model)

model = Model(inputs, model)
model.compile(loss=univariate_gaussian_loss, optimizer=OPTIMIZER)
model.summary()

callbacks = [
    TensorBoard(log_dir='./tensorboard_log/' + TIMESTAMP + ' ' + SCRIPT_NAME, write_graph=False),
    DecypherAll(lambda x: str(x)),
    EarlyStopping(patience=40, min_delta=1e-4)
]

history = model.fit(x=training_vectors,
                    y=target_vectors,
                    epochs=EPOCHS,
                    batch_size=BATCH_SIZE,
                    validation_split=TRAIN_VALIDATE_SPLIT,
                    callbacks=callbacks).history

notify(TIMESTAMP, SCRIPT_NAME, 'validation loss of ' + str(history['val_loss'][-1]))
print(SCRIPT_NAME, 'finished successfully')
