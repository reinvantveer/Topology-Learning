from keras import backend as K

from topoml_util.GeoVectorizer import RENDER_LEN, GEOM_TYPE_LEN, ONE_HOT_LEN
from topoml_util.gaussian_loss import bivariate_gaussian, univariate_gaussian


class GaussianMixtureLoss:
    def __init__(self, num_components, num_points):
        self.num_points = num_points
        self.num_components = num_components

    def geom_gaussian_mixture_loss(self, y_true, y_pred):
        """
        Calculates a loss from a rank 3 sequence, representing a self.num_components * 6 slice (the mixture components)
        plus one-hot encoded sequences of geometry type (8) and render/stop action type (3)
        :param y_true: rank 3 of shape(records, points, true_point_features >= 17) truth values tensor
        :param y_pred: rank 3 of shape(records, points, pred_point_features >= 17) predicted values tensor
        :return: a summed mixture loss and categorical cross entropy losses for the geometry type and stop bits
        """
        # loss fn based on eq #26 of http://arxiv.org/abs/1308.0850.
        # Reshape to one target component to be broadcasted over self.num_components
        true_coordinates = y_true[..., :2]
        # It would be nice to be able to do
        # shape = [*y_true.shape[:-1], 1, 2]
        shape = [-1, self.num_points, 1, 2]
        true_coordinates = K.reshape(true_coordinates, tuple(shape))

        y_pred_gmm_components = y_pred[..., :-ONE_HOT_LEN]
        predicted_components = K.reshape(
            y_pred_gmm_components,
            # (*y_pred.shape[:-1], -1, 6))  # This would be nice
            (-1, self.num_points, self.num_components, 6))

        pi_index = 5  # mixture component weight
        pi_weights = K.softmax(predicted_components[..., pi_index])
        gmm = bivariate_gaussian(true_coordinates, predicted_components) * pi_weights
        gmm_loss = K.sum(-K.log(gmm + K.epsilon()))

        render_action = K.softmax(y_true[..., -RENDER_LEN:])
        neg_full_stop_chance = 1 - render_action[..., 2]  # 1 minus the chance of full stop
        gmm_loss = gmm_loss * neg_full_stop_chance

        geom_type_error = K.categorical_crossentropy(
            K.softmax(y_true[..., -(GEOM_TYPE_LEN + RENDER_LEN - 1):-RENDER_LEN]),
            K.softmax(y_pred[..., -(GEOM_TYPE_LEN + RENDER_LEN - 1):-RENDER_LEN]))
        render_error = K.categorical_crossentropy(
            K.softmax(y_true[..., -RENDER_LEN:]),
            K.softmax(y_pred[..., -RENDER_LEN:]))

        return gmm_loss + geom_type_error + render_error

    def univariate_gmm_loss(self, true, pred):
        """
        A simple loss function for rank 3 single gaussian mixture models
        :param true: truth values tensor
        :param pred: prediction values tensor
        :return: loss values tensor
        """
        if not true.shape == pred.shape:
            print(
                'Warning: truth', true.shape, 'and prediction tensors', pred.shape, 'do not have the same shape. The '
                'outcome of the loss function may be unpredictable.')

        # true_components = K.reshape(true, (-1, self.num_components, 3))
        # TODO: make reshape op rank agnostic
        predicted_components = K.reshape(pred, (-1, self.num_components, 3))

        pi_index = 2
        pi_weights = K.softmax(pred[..., pi_index])
        gmm = univariate_gaussian(true, predicted_components) * pi_weights
        gmm_loss = -K.log(K.sum(gmm + K.epsilon()))

        return gmm_loss
