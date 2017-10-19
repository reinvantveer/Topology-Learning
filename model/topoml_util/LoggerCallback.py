import pprint
from keras.callbacks import Callback
import random
from datetime import datetime
import numpy as np

pp = pprint.PrettyPrinter()


class EpochLogger(Callback):
    def __init__(self, input_func, target_func, predict_func, aggregate_func, sample_size=3, stdout=False):
        super().__init__()
        self.input_func = input_func
        self.target_func = target_func
        self.predict_func = predict_func
        self.aggregate_func = aggregate_func
        self.sample_size = sample_size
        self.log_to_stdout = stdout

    def on_epoch_end(self, epoch, logs=None):
        random.seed(datetime.now())
        validation_samples = random.sample(range(len(self.validation_data[0])), self.sample_size)
        input_samples = [self.validation_data[0][sample] for sample in validation_samples]
        target_samples = [self.validation_data[1][sample] for sample in validation_samples]
        predictions = self.model.predict(np.array(input_samples))

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
