import os
import pprint
from shapely.geometry import Point
from topoml_util.wkt2pyplot import wkt2pyplot, save_plot
from keras.callbacks import Callback
import random
from datetime import datetime
import numpy as np

from topoml_util.GeoVectorizer import GeoVectorizer

pp = pprint.PrettyPrinter()


class DecypherAll(Callback):
    def __init__(self, gmm_size=1, sample_size=3, input_slice=lambda x: x[0:1], target_slice=lambda x: x[1:2], stdout=False, plot_dir='plots'):
        """
        Class constructor that instantiates with a few vital settings in order to decypher the output
        :type target_slice: object
        :param gmm_size: size as an integer of the gaussian mixture model
        :param sample_size: size as an integer of the number of samples to log
        :param stdout: boolean whether or not to log to stdout. Mixture models can have a lot of output.
        :param plot_dir: string of a directory to save plots to, relative to the path called to execute the script
        """
        super().__init__()
        self.gmm_size = gmm_size
        self.sample_size = sample_size
        self.input_slice = input_slice
        self.target_slice = target_slice
        self.stdout = stdout

        os.makedirs(plot_dir, exist_ok=True)
        self.plot_dir = plot_dir

    def on_epoch_end(self, epoch, logs=None):
        """
        Epochal logging function that outputs to a pyplot saved to a timestamped .png file
        :param epoch: automatically instantiated by Keras
        :param logs: automatically instantiated by Keras
        """
        random.seed(datetime.now())

        sample_indexes = random.sample(range(len(self.validation_data[0])), self.sample_size)
        inputs = np.array(self.input_slice(self.validation_data))
        targets = np.array(self.target_slice(self.validation_data))
        input_samples = [inputs[:, sample_index] for sample_index in sample_indexes]
        target_samples = [targets[:, sample_index] for sample_index in sample_indexes]

        predictions = []
        for sample_index in sample_indexes:
            sample = inputs[:, sample_index:sample_index + 1]
            predictions.append(self.model.predict([*sample]))

        print('\nPlotting output for %i inputs, targets and predictions...' % len(predictions))

        for (input_vectors, target_vectors, prediction_vectors) in zip(input_samples, target_samples, predictions):
            timestamp = str(datetime.now()).replace(':', '.')

            if self.stdout:
                print('Input:')
                pp.pprint(input_vectors)
                print('Target:')
                pp.pprint(target_vectors)
                print('Prediction:')
                pp.pprint(prediction_vectors)

            input_polys = [GeoVectorizer.decypher(poly) for poly in input_vectors]
            target_polys = [GeoVectorizer.decypher(target_vectors[0])]
            prediction_points = [Point(point).wkt for point in
                                 GeoVectorizer(gmm_size=self.gmm_size).decypher_gmm_geom(prediction_vectors[0], 500)]

            geoms = input_polys, target_polys, prediction_points
            save_plot(self.plot_dir, geoms, timestamp)
