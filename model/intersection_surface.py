import os
from datetime import datetime

import numpy as np
from keras import Input
from keras.callbacks import TensorBoard, EarlyStopping
from keras.engine import Model
from keras.layers import LSTM, Dense, concatenate, RepeatVector
from keras.optimizers import Adam
from topoml_util.LoggerCallback import EpochLogger
from topoml_util.gaussian_loss import univariate_gaussian_loss
from topoml_util.geom_scaler import localized_normal, localized_mean
from topoml_util.slack_send import notify
from topoml_util.wkt2pyplot import wkt2pyplot

# To suppress tensorflow info level messages:
# export TF_CPP_MIN_LOG_LEVEL=2

SCRIPT_VERSION = "0.0.4"
SCRIPT_NAME = os.path.basename(__file__)
TIMESTAMP = str(datetime.now()).replace(':', '.')
SIGNATURE = SCRIPT_NAME + ' ' + TIMESTAMP
DATA_FILE = '../files/geodata_vectorized.npz'
BATCH_SIZE = 1024
TRAIN_VALIDATE_SPLIT = 0.1
HIDDEN_SIZE = 128
REPEAT_HIDDEN = 3
EPOCHS = 400
OPTIMIZER = Adam(lr=1e-3)

loaded = np.load(DATA_FILE)
raw_brt_vectors = loaded['brt_vectors']
raw_osm_vectors = loaded['osm_vectors']
raw_target_vectors = loaded['intersection_surface'][:, 0, :]

brt_vectors = []
osm_vectors = []
target_vectors = []

# skip non-intersecting geometries
for brt, osm, target in zip(raw_brt_vectors, raw_osm_vectors, raw_target_vectors):
    if not target[0] == 0:  # a zero coordinate designates an empty geometry
        brt_vectors.append(brt)
        osm_vectors.append(osm)
        target_vectors.append(target)

# data whitening
means = localized_mean(brt_vectors)
brt_vectors = localized_normal(brt_vectors, means, 1e4)
osm_vectors = localized_normal(osm_vectors, means, 1e4)
target_vectors = np.array(target_vectors)

# shape determination
(data_points, brt_max_points, BRT_INPUT_VECTOR_LEN) = brt_vectors.shape
(_, osm_max_points, OSM_INPUT_VECTOR_LEN) = osm_vectors.shape
target_max_points = target_vectors.shape[1]
output_seq_length = 2
output_size_2d = target_max_points * output_seq_length

brt_inputs = Input(shape=(brt_max_points, BRT_INPUT_VECTOR_LEN))
brt_model = LSTM(brt_max_points * 2, activation='relu')(brt_inputs)

osm_inputs = Input(shape=(osm_max_points, OSM_INPUT_VECTOR_LEN))
osm_model = LSTM(osm_max_points * 2, activation='relu')(osm_inputs)

concat = concatenate([brt_model, osm_model])
model = RepeatVector(target_max_points)(concat)

for layer in range(REPEAT_HIDDEN):
    model = LSTM(HIDDEN_SIZE, activation='relu', return_sequences=True)(model)
    model = Dense(32, activation='relu')(model)

model = LSTM(HIDDEN_SIZE, activation='relu')(model)  # Flatten
model = Dense(2)(model)
model = Model(inputs=[brt_inputs, osm_inputs], outputs=model)
model.compile(loss=univariate_gaussian_loss, optimizer=OPTIMIZER)
model.summary()

callbacks = [
    TensorBoard(log_dir='./tensorboard_log/' + TIMESTAMP + ' ' + SCRIPT_NAME, write_graph=False),
    EpochLogger(
        input_func=lambda x: None,
        target_func=lambda x: None,
        predict_func=lambda x: None,
        aggregate_func=lambda x: None,
        input_slice=lambda x: x[0:2],
        target_slice=lambda x: x[2:3],
        stdout=True),
    EarlyStopping(patience=40, min_delta=0.001)
]

history = model.fit(
    x=[brt_vectors, osm_vectors],
    y=target_vectors,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=TRAIN_VALIDATE_SPLIT,
    callbacks=callbacks).history

prediction = model.predict([brt_vectors[0:1000], osm_vectors[0:1000]])
intersecting_error = np.abs(prediction[:, 0] - target_vectors[0:1000, 0])

zipped = zip(target_vectors[0:1000, 0], prediction[:, 0])
sorted_results = sorted(zipped, key=lambda record: abs(record[1] - record[0]))

print('Intersection surface area mean:', np.mean(target_vectors))
print('Intersecting error mean:', np.mean(intersecting_error))

plot_samples = 50

if False:
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
