from topoml_util.gaussian_loss import r4_bivariate_gaussian
import numpy as np

class GaussianMixtureLoss:
    def __init__(self, num_components):
        self.num_components = num_components
        
    @staticmethod
    def softmax(x):
        """Compute softmax values for each sets of scores in x."""
        return np.exp(x) / np.sum(np.exp(x), axis=0)

    @staticmethod
    def epsilon():
        return 1e-16

    def geom_gaussian_mixture_loss(self, y_true, y_pred):
        # loss fn based on eq #26 of http://arxiv.org/abs/1308.0850.
        (data_points, points, features) = y_pred.shape
        geom_type_index = 6 * self.num_components  # Calculate offset from parameters times components
        render_index = geom_type_index + 8
        pi_index = 5

        predicted_components = np.reshape(y_pred[:geom_type_index], (-1, points.value, self.num_components, 6))
        pi = self.softmax(predicted_components[:, :, :, pi_index])

        true_components = np.reshape(y_true[:geom_type_index], (-1, points.value, self.num_components, 6))

        gmm = r4_bivariate_gaussian(true_components, predicted_components) * pi
        gmm_loss = np.sum(-np.log(gmm + self.epsilon()))

        # Zero out loss terms beyond N_s, the last actual stroke
        render = 1 - np.mean(y_pred[:, :, render_index:render_index + 2])  # RENDER and STOP values

        gmm_loss = gmm_loss * render

        return gmm_loss


