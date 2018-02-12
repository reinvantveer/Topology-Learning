import os
from datetime import datetime

import numpy as np
import sys
from keras import Input
from keras.callbacks import TensorBoard, EarlyStopping
from keras.engine import Model
from keras.layers import LSTM, TimeDistributed, Dense, Flatten
from keras.optimizers import Adam
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from topoml_util import geom_scaler
from topoml_util.slack_send import notify

SCRIPT_VERSION = '1.0.0'
SCRIPT_NAME = os.path.basename(__file__)
TIMESTAMP = str(datetime.now()).replace(':', '.')
SIGNATURE = SCRIPT_NAME + ' ' + TIMESTAMP
TRAINING_DATA_FILE = '../files/neighborhoods/neighborhoods_train.npz'

# Hyperparameters
BATCH_SIZE = int(os.getenv('BATCH_SIZE', 384))
TRAIN_VALIDATE_SPLIT = float(os.getenv('TRAIN_VALIDATE_SPLIT', 0.1))
REPEAT_DEEP_ARCH = int(os.getenv('REPEAT_DEEP_ARCH', 0))
LSTM_SIZE = int(os.getenv('LSTM_SIZE', 256))
DENSE_SIZE = int(os.getenv('DENSE_SIZE', 64))
EPOCHS = int(os.getenv('EPOCHS', 200))
LEARNING_RATE = float(os.getenv('LEARNING_RATE', 1e-4))
PATIENCE = int(os.getenv('PATIENCE', 16))
RECURRENT_DROPOUT = float(os.getenv('RECURRENT_DROPOUT', 0.05))
GEOM_SCALE = float(os.getenv('GEOM_SCALE', 0))  # If no default or 0: overridden when data is known
OPTIMIZER = Adam(lr=LEARNING_RATE)

# Load and normalize data
train_loaded = np.load(TRAINING_DATA_FILE)
train_geoms = train_loaded['input_geoms']
train_above_or_below_median = train_loaded['above_or_below_median']

# Determine final test mode or standard
if sys.argv[1] in ['-t', '--test']:
    print('Training in final test mode')
    TEST_DATA_FILE = '../files/neighborhoods/neighborhoods_test.npz'
    test_loaded = np.load(TEST_DATA_FILE)
    test_geoms = test_loaded['input_geoms']
    test_above_or_below_median = test_loaded['above_or_below_median'][:, 0]
else:
    print('Training in standard validation mode')
    # Split the training data in random seen/unseen sets
    train_geoms, test_geoms, train_above_or_below_median, test_above_or_below_median = train_test_split(
        train_geoms, train_above_or_below_median, test_size=0.1)

# Normalize
geom_scale = GEOM_SCALE or geom_scaler.scale(train_geoms)
train_geoms = geom_scaler.transform(train_geoms, geom_scale)
test_geoms = geom_scaler.transform(test_geoms, geom_scale)  # re-use variance from training


message = '''
running {} with 
version: {}                batch size: {} 
train/validate split: {}   repeat deep: {} 
lstm size: {}              dense size: {} 
epochs: {}                 learning rate: {}
geometry scale: {:f}       recurrent dropout: {}
patience {}
'''.format(
    SIGNATURE,
    SCRIPT_VERSION,         BATCH_SIZE,
    TRAIN_VALIDATE_SPLIT,   REPEAT_DEEP_ARCH,
    LSTM_SIZE,              DENSE_SIZE,
    EPOCHS,                 LEARNING_RATE,
    geom_scale,             RECURRENT_DROPOUT,
    PATIENCE,
)
print(message)

# Shape determination
geom_max_points, geom_vector_len = train_geoms.shape[1:]
output_size = train_above_or_below_median.shape[-1]

# Build model
inputs = Input(shape=(geom_max_points, geom_vector_len))
model = LSTM(LSTM_SIZE, return_sequences=True, recurrent_dropout=RECURRENT_DROPOUT)(inputs)
model = TimeDistributed(Dense(DENSE_SIZE, activation='relu'))(model)

for layer in range(REPEAT_DEEP_ARCH):
    model = LSTM(LSTM_SIZE, return_sequences=True, recurrent_dropout=RECURRENT_DROPOUT)(model)

model = Dense(DENSE_SIZE, activation='relu')(model)
model = Flatten()(model)
model = Dense(output_size, activation='softmax')(model)

model = Model(inputs=inputs, outputs=model)
model.compile(
    loss='categorical_crossentropy',
    metrics=['accuracy'],
    optimizer=OPTIMIZER),
model.summary()

# Callbacks
callbacks = [
    TensorBoard(log_dir='./tensorboard_log/' + SIGNATURE, write_graph=False),
    EarlyStopping(patience=PATIENCE, min_delta=0.001)
]

history = model.fit(
    x=train_geoms,
    y=train_above_or_below_median,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=TRAIN_VALIDATE_SPLIT,
    callbacks=callbacks).history

# Run on unseen test data
test_pred = [np.argmax(prediction) for prediction in model.predict(test_geoms)]
accuracy = accuracy_score(test_above_or_below_median, test_pred)
message = '''
test accuracy of {:f} with 
version: {}                batch size {} 
train/validate split {}    repeat deep arch {} 
lstm size {}               dense size {} 
epochs {}                  learning rate {}
geometry scale {:f}        recurrent dropout {}
patience {}
'''.format(
    accuracy,          SCRIPT_VERSION,
    BATCH_SIZE,             TRAIN_VALIDATE_SPLIT,
    REPEAT_DEEP_ARCH,       LSTM_SIZE,
    DENSE_SIZE,             len(history['val_loss']),
    LEARNING_RATE,          geom_scale,
    RECURRENT_DROPOUT,      PATIENCE,
)

notify(SIGNATURE, message)
print(SCRIPT_NAME, 'finished successfully')
