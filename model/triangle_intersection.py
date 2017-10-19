import os
from datetime import datetime
from shapely.wkt import loads
from shapely.geometry import Polygon
import numpy as np
from keras import Input
from keras.callbacks import TensorBoard
from keras.engine import Model
from keras.layers import LSTM, Dense
from keras.optimizers import Adam
from shutil import copyfile

from topoml_util.LoggerCallback import EpochLogger
from topoml_util.GeoVectorizer import GeoVectorizer
from topoml_util.GaussianMixtureLoss import GaussianMixtureLoss
from topoml_util.slack_send import notify
from topoml_util.wkt2pyplot import wkt2pyplot

SCRIPT_NAME = os.path.basename(__file__)
TIMESTAMP = str(datetime.now()).replace(':', '.')
DATA_FILE = '../files/triangles.npz'
BATCH_SIZE = 2048
TRAIN_VALIDATE_SPLIT = 0.1
LATENT_SIZE = 64
EPOCHS = 400
OPTIMIZER = Adam(lr=1e-3, decay=2e-5)

# Archive the configuration
copyfile(__file__, 'configs/' + TIMESTAMP + ' ' + SCRIPT_NAME)

loaded = np.load(DATA_FILE)
training_vectors = loaded['point_sequence']
target_vectors = loaded['intersection_geoms']
(set_size, _) = training_vectors.shape
training_vectors = np.reshape(training_vectors, (set_size, 1, 12))
target_triangles = [GeoVectorizer.vectorize_wkt(triangle, 6)
                    for triangle in training_vectors]
# triangles = [GeoVectorizer.interpolate(point_sequence, len(point_sequence) * 20)
#                         for point_sequence in triangles]
(_, max_points, GEO_VECTOR_LEN) = np.array(target_vectors).shape

inputs = Input(shape=training_vectors.shape[1:])
model = LSTM(LATENT_SIZE, activation='relu', return_sequences=True)(inputs)
model = Dense(32, activation='relu')(model)
model = LSTM(LATENT_SIZE, activation='relu', return_sequences=True)(model)
model = Dense(32, activation='relu')(model)
model = Dense(GEO_VECTOR_LEN)(model)
model = Model(inputs, model)

loss = GaussianMixtureLoss(num_points=max_points, num_components=1)
model.compile(loss=GaussianMixtureLoss.geom_gaussian_mixture_loss, optimizer=OPTIMIZER)
model.summary()

callbacks = [
    TensorBoard(log_dir='./tensorboard_log/' + TIMESTAMP, write_graph=False),
    EpochLogger(
        input_func=GeoVectorizer.decypher,
        target_func=GeoVectorizer.decypher,
        predict_func=GeoVectorizer.decypher,
        aggregate_func=wkt2pyplot,
        stdout=True)
]

history = model.fit(
    x=training_vectors,
    y=target_vectors,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=TRAIN_VALIDATE_SPLIT,
    callbacks=callbacks).history

plot_sample = training_vectors[-10000:]
prediction = model.predict(plot_sample)
intersecting_error = np.abs(prediction[:, 0] - target_vectors[-10000:])

triangle_vectors = plot_sample.reshape(10000, 6, 2)
training_triangles = np.array([[Polygon(point_set[0:3]).wkt, Polygon(point_set[3:]).wkt]
                               for point_set in triangle_vectors])

notify(TIMESTAMP, SCRIPT_NAME, 'validation loss of ' + str(history['val_loss'][-1]))
print(SCRIPT_NAME, 'finished successfully')
