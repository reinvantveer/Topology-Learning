from datetime import datetime
import numpy as np
import os
from keras import Input
from keras.callbacks import TensorBoard, EarlyStopping
from keras.engine import Model
from keras.layers import Dense, LSTM
from keras.optimizers import Adam

from topoml_util.ConsoleLogger import DecypherAll
from topoml_util.geom_loss import bivariate_gaussian_loss
from topoml_util.slack_send import notify

TIMESTAMP = str(datetime.now()).replace(':', '.')
SCRIPT_NAME = os.path.basename(__file__)
EPOCHS = 200
BATCH_SIZE = 512
TRAINING_SIZE = 100000
TRAIN_VALIDATE_SPLIT = 0.2

input_2d = np.repeat([[[0.2, 15, 0, 0, 0]]], 11, axis=1)
input_2d = np.repeat(input_2d, TRAINING_SIZE, axis=0)
(data_points, max_points, vector_len) = input_2d.shape

inputs = Input(name='Input', shape=(max_points, vector_len))
model = LSTM(vector_len, return_sequences=True)(inputs)
# The dense layer is required for input values exceeding 1e0
model = Dense(vector_len)(model)
model = Model(inputs, model)
model.compile(
    loss=bivariate_gaussian_loss,
    metrics='mse',
    optimizer=Adam(lr=0.01))
model.summary()

callbacks = [
    TensorBoard(log_dir='./tensorboard_log/' + TIMESTAMP, write_graph=False),
    DecypherAll(lambda x: str(x)),
    EarlyStopping(patience=20)
]

history = model.fit(x=input_2d,
          y=input_2d,
          epochs=EPOCHS,
          batch_size=BATCH_SIZE,
          validation_split=TRAIN_VALIDATE_SPLIT,
          callbacks=callbacks).history

notify(TIMESTAMP, SCRIPT_NAME, history[-1])
print('Done!')
