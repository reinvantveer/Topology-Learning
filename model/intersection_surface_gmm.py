import os
from datetime import datetime

import numpy as np
from keras import Input
from keras.callbacks import TensorBoard
from keras.engine import Model
from keras.layers import LSTM, Dense, Reshape
from keras.optimizers import Adam

from topoml_util.GaussianMixtureLoss import GaussianMixtureLoss
from topoml_util.LoggerCallback import EpochLogger
from topoml_util.geom_scaler import localized_normal, localized_mean
from topoml_util.GeoVectorizer import GeoVectorizer
from topoml_util.wkt2pyplot import wkt2pyplot
from topoml_util.slack_send import notify

SCRIPT_VERSION = "0.0.1"
SCRIPT_NAME = os.path.basename(__file__)
TIMESTAMP = str(datetime.now()).replace(':', '.')  # for Windows compat
SIGNATURE = SCRIPT_NAME + ' ' + TIMESTAMP
DATA_FILE = '../files/geodata_vectorized.npz'
BATCH_SIZE = 5122
TRAIN_VALIDATE_SPLIT = 0.1
LATENT_SIZE = 128
EPOCHS = 50
OPTIMIZER = Adam(lr=1e-3)

NUM_COMPONENTS = 10

loaded = np.load(DATA_FILE)
raw_training_vectors = loaded['input_geoms']
raw_target_vectors = loaded['intersection_surface'][:, 0, :]

training_vectors = []
target_vectors = []

for index, target in enumerate(raw_target_vectors):
    if not target[0] == 0:  # a zero coordinate designates an empty geometry
        training_vectors.append(raw_training_vectors[index])
        target_vectors.append(raw_target_vectors[index])


means = localized_mean(training_vectors)
training_vectors = localized_normal(training_vectors, means, 1e4)
(data_points, max_points, GEO_VECTOR_LEN) = training_vectors.shape

target_vectors = np.reshape(target_vectors, (data_points, 1, 2))

inputs = Input(shape=(max_points, GEO_VECTOR_LEN))
model = Dense(64)(inputs)
model = LSTM(LATENT_SIZE, activation='relu')(model)
model = Dense(NUM_COMPONENTS * 3)(model)
model = Reshape((1, NUM_COMPONENTS * 3))(model)
model = Model(inputs, model)

GMM = GaussianMixtureLoss(NUM_COMPONENTS, max_points)
model.compile(loss=GMM.univariate_gmm_loss, optimizer=OPTIMIZER)
model.summary()

tb_callback = TensorBoard(log_dir='./tensorboard_log/' + TIMESTAMP, write_graph=False)
epoch_callback = EpochLogger(
    input_func=GeoVectorizer.decypher,
    target_func=lambda x: str(x),
    predict_func=lambda x: str(x),
    aggregate_func=None,
    stdout=True
)

history = model.fit(
    x=training_vectors,
    y=np.array(target_vectors),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=TRAIN_VALIDATE_SPLIT,
    callbacks=[epoch_callback, tb_callback]).history

prediction = model.predict(training_vectors[0:1000])
intersecting_error = np.abs(prediction[:, 0] - target_vectors[0:1000, 0])

decyphered = [GeoVectorizer.decypher(vector).split('\n') for vector in training_vectors[0:1000]]
zipped = zip(decyphered, target_vectors[0:1000, 0], prediction[:, 0])
sorted_results = sorted(zipped, key=lambda record: abs(record[2] - record[1]))

print('Intersection surface area mean:', np.mean(target_vectors))
print('Intersecting error mean:', np.mean(intersecting_error))

plot_samples = 50
print('Saving top and bottom', plot_samples, 'results as plots, this will take a few minutes...')
# print('Worst', plot_samples, 'results: ', sorted_results[-plot_samples:])
for result in sorted_results[-plot_samples:]:
    timestamp = str(datetime.now()).replace(':', '.')
    plot, _, ax = wkt2pyplot(result[0])
    plot.text(0.01, 0.06, 'target: ' + str(result[1]), transform=ax.transAxes)
    plot.text(0.01, 0.01, 'prediction: ' + str(result[2]), transform=ax.transAxes)
    plot.savefig('./plot_images/bad_' + timestamp + '.png')
    plot.close()

# print('Best', plot_samples, 'results:', sorted_results[0:plot_samples])
for result in sorted_results[0:plot_samples]:
    timestamp = str(datetime.now()).replace(':', '.')
    plot, _, ax = wkt2pyplot(result[0])
    plot.text(0.01, 0.06, 'target: ' + str(result[1]), transform=ax.transAxes)
    plot.text(0.01, 0.01, 'prediction: ' + str(result[2]), transform=ax.transAxes)
    plot.savefig('./plot_images/good_' + timestamp + '.png')
    plot.close()

notify(TIMESTAMP, SCRIPT_NAME, 'validation loss of ' + str(history['val_loss'][-1]))
print(SCRIPT_NAME, 'finished successfully')
