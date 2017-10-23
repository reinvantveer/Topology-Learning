import pprint
from keras.callbacks import Callback
import random
from datetime import datetime
import numpy as np

pp = pprint.PrettyPrinter()


class EpochLogger(Callback):
    def __init__(self, input_func, target_func, predict_func, aggregate_func, sample_size=3, stdout=False,
                 input_slice=lambda x: x[0:1], target_slice=lambda x: x[1:2]):
        super().__init__()
        self.input_func = input_func
        self.target_func = target_func
        self.predict_func = predict_func
        self.aggregate_func = aggregate_func
        self.sample_size = sample_size
        self.log_to_stdout = stdout
        self.input_slice = input_slice
        self.target_slice = target_slice

    def on_epoch_end(self, epoch, logs=None):
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

        print('\nLogging output for %i inputs, targets and predictions...' % len(predictions))

        for (input, target, prediction) in zip(input_samples, target_samples, predictions):

            if self.log_to_stdout:
                print('Input:')
                pp.pprint(input)
                print('Target:')
                pp.pprint(target)
                print('Prediction:')
                pp.pprint(prediction)
                print('')

            if self.aggregate_func:
                self.aggregate_func(
                    (self.input_func(input), self.target_func(target), self.predict_func(prediction)))
