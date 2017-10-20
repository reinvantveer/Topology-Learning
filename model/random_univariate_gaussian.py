from datetime import datetime
import numpy as np
import os
from keras import Input
from keras.callbacks import TensorBoard, EarlyStopping
from keras.engine import Model
from keras.layers import LSTM, Dense
from keras.optimizers import Adam
from shutil import copyfile

from topoml_util.ConsoleLogger import DecypherAll
from topoml_util.gaussian_loss import univariate_gaussian_loss
from topoml_util.slack_send import notify

SCRIPT_VERSION = "0.0.3"
TIMESTAMP = str(datetime.now()).replace(':', '.')
SCRIPT_NAME = os.path.basename(__file__)
EPOCHS = 200
BATCH_SIZE = 100
TRAINING_SIZE = 50000
TRAIN_VALIDATE_SPLIT = 0.2

# Archive the configuration
copyfile(__file__, 'configs/' + TIMESTAMP + ' ' + SCRIPT_NAME)

univariate = np.random.randint(low=1, high=20, size=(TRAINING_SIZE, 1, 1))
(_, max_points, SEQ_LEN) = univariate.shape

inputs = Input(name='Input', shape=(max_points, SEQ_LEN))
model = LSTM(SEQ_LEN, activation='relu', return_sequences=True)(inputs)
model = Dense(2)(model)
model = Model(inputs, model)
model.compile(loss=univariate_gaussian_loss, optimizer=Adam(lr=0.001))
model.summary()

callbacks = [
    TensorBoard(log_dir='./tensorboard_log/' + TIMESTAMP + ' ' + SCRIPT_NAME, write_graph=False),
    DecypherAll(lambda x: str(x)),
    EarlyStopping(patience=40, min_delta=1e-4)
]

history = model.fit(x=univariate,
                    y=univariate,
                    epochs=EPOCHS,
                    batch_size=BATCH_SIZE,
                    validation_split=TRAIN_VALIDATE_SPLIT,
                    callbacks=callbacks).history

notify(TIMESTAMP, SCRIPT_NAME, 'validation loss of ' + str(history['val_loss'][-1]))
print(SCRIPT_NAME, 'finished successfully')
