from datetime import datetime
import numpy as np
import os
from keras import Input
from keras.callbacks import TensorBoard, EarlyStopping
from keras.engine import Model
from keras.layers import LSTM, TimeDistributed, Dense
from keras.optimizers import Adam
from shutil import copyfile

from topoml_util.ConsoleLogger import DecypherAll
from topoml_util.geom_loss import bivariate_gaussian_loss
from topoml_util.slack_send import notify

TIMESTAMP = str(datetime.now()).replace(':', '.')
SCRIPT_NAME = os.path.basename(__file__)
EPOCHS = 200
BATCH_SIZE = 100
TRAINING_SIZE = 50000
TRAIN_VALIDATE_SPLIT = 0.2

# Archive the configuration
copyfile(__file__, 'configs/' + TIMESTAMP + ' ' + SCRIPT_NAME)

bivariate = np.random.randint(low=1, high=20, size=(TRAINING_SIZE, 1, 5))
(_, max_points, SEQ_LEN) = bivariate.shape

inputs = Input(name='Input', shape=(max_points, SEQ_LEN))
model = LSTM(SEQ_LEN, return_sequences=True)(inputs)
model = TimeDistributed(Dense(5))(model)
model = Model(inputs, model)
model.compile(loss=bivariate_gaussian_loss, optimizer=Adam(lr=0.001))
model.summary()

callbacks = [
    DecypherAll(lambda x: str(x)),
    TensorBoard(log_dir='./tensorboard_log/' + TIMESTAMP + ' ' + SCRIPT_NAME, write_graph=False),
    EarlyStopping(patience=40, min_delta=1e-3)
]

history = model.fit(x=bivariate,
                    y=bivariate,
                    epochs=EPOCHS,
                    batch_size=BATCH_SIZE,
                    validation_split=TRAIN_VALIDATE_SPLIT,
                    callbacks=callbacks).history

notify(TIMESTAMP, SCRIPT_NAME, 'validation loss of ' + str(history['val_loss'][-1]))
print('Done!')
