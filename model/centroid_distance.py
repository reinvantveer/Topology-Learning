from datetime import datetime
import os
import numpy as np
from keras import Input
from keras.callbacks import TensorBoard
from keras.engine import Model
from keras.layers import Dense, Flatten
from keras.optimizers import Adam
from shutil import copyfile

from topoml_util.ConsoleLogger import DecypherAll
from topoml_util.gaussian_loss import univariate_gaussian_loss

# To suppress tensorflow info level messages:
# export TF_CPP_MIN_LOG_LEVEL=2
from topoml_util.slack_send import notify

SCRIPT_NAME = os.path.basename(__file__)
TIMESTAMP = str(datetime.now()).replace(':', '.')
DATA_FILE = '../files/geodata_vectorized.npz'
BATCH_SIZE = 1024
TRAIN_VALIDATE_SPLIT = 0.1
LATENT_SIZE = 16
EPOCHS = 50
OPTIMIZER = Adam(lr=0.001)

# Archive the configuration
copyfile(__file__, 'configs/' + TIMESTAMP + ' ' + SCRIPT_NAME)

loaded = np.load(DATA_FILE)
training_vectors = loaded['centroids'][:, :, 0:2]

# Bring coordinates and distance in roughly the same scale
base_precision = 1e4
base = np.floor(base_precision * training_vectors[:, 0:1, :])
base = np.repeat(base, 2, axis=1)
training_vectors = (base_precision * training_vectors) - base

(data_points, max_points, GEO_VECTOR_LEN) = training_vectors.shape
target_vectors = loaded['centroid_distance'][:, 0, :]

inputs = Input(name='Input', shape=(max_points, GEO_VECTOR_LEN))
# This can be a simple Dense layer of size 16 as well
model = Flatten()(inputs)
model = Dense(LATENT_SIZE, activation='relu')(model)
model = Dense(2)(model)

model = Model(inputs, model)
model.compile(loss=univariate_gaussian_loss, optimizer=OPTIMIZER)
model.summary()

tb_callback = TensorBoard(log_dir='./tensorboard_log/' + TIMESTAMP, write_graph=False)
my_callback = DecypherAll(lambda x: str(x))

history = model.fit(x=training_vectors,
                    y=target_vectors,
                    epochs=EPOCHS,
                    batch_size=BATCH_SIZE,
                    validation_split=TRAIN_VALIDATE_SPLIT,
                    callbacks=[my_callback, tb_callback]).history

prediction = model.predict(training_vectors[0:1000])
error = np.sum(np.abs(prediction[:, 0] - target_vectors[0:1000, 0])) / 1000
print('Error factor:', error)

notify(TIMESTAMP, SCRIPT_NAME, 'validation loss of ' + str(history['val_loss'][-1]))
print(SCRIPT_NAME, 'finished successfully')
