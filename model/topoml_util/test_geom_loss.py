import unittest
import numpy as np
import tensorflow as tf
from keras import backend as K

from .gaussian_loss import geom_gaussian_loss, bivariate_gaussian_loss, univariate_gaussian_loss

PRECISION = 6
sess = tf.InteractiveSession()


class TestGeomLossFunction(unittest.TestCase):
    def test_zero_loss(self):
        tensor1 = np.array([[[1, 2, 0, 0, 0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]]], dtype=float)
        tensor2 = np.array([[[1, 2, 0, 0, 0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]]], dtype=float)
        loss = geom_gaussian_loss(tensor1, tensor2).eval()
        self.assertEqual(loss, 1.87002731)

    def test_geom_type_loss(self):
        tensor1 = np.array([[[1, 2, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]], dtype=float)
        tensor2 = np.array([[[1, 2, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]]], dtype=float)
        K.print_tensor(tensor1)
        loss = geom_gaussian_loss(tensor1, tensor2).eval()
        self.assertEqual(loss, 3.26972228)

    def test_geom_type_and_render_op_loss(self):
        tensor1 = np.array([[[1, 2, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]]], dtype=float)
        tensor2 = np.array([[[1, 2, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1]]], dtype=float)
        loss = geom_gaussian_loss(tensor1, tensor2).eval()
        self.assertEqual(loss, 3.51061315)

    def test_geom_one_coordinate_loss(self):
        tensor1 = np.array([[[1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]], dtype=float)
        tensor2 = np.array([[[1, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]], dtype=float)
        loss = geom_gaussian_loss(tensor1, tensor2).eval()
        self.assertEqual(loss, 3.67805383)

    def test_geom_double_coordinate_loss(self):
        tensor1 = np.array([[[1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]], dtype=float)
        tensor2 = np.array([[[5, 52, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]], dtype=float)
        loss = geom_gaussian_loss(tensor1, tensor2).eval()
        self.assertEqual(loss[0, 0], 1261.17805383)


class TestGaussian2dLoss(unittest.TestCase):
    def test_2d_gaussian_zeros(self):
        target = np.array([[[0, 0]]], dtype=float)
        prediction = np.array([[[0, 0, 0, 0, 0]]], dtype=float)
        loss = bivariate_gaussian_loss(target, prediction).eval()
        self.assertAlmostEqual(loss[0, 0], 1.1048509233685306, places=PRECISION)

    def test_2d_gaussian_small_mu_diff(self):
        target = np.array([[[5, 52]]], dtype=float)
        prediction = np.array([[[5 + 1e-6, 52 + 1e-6, 0, 0, 0]]], dtype=float)
        loss = bivariate_gaussian_loss(target, prediction).eval()
        self.assertAlmostEqual(loss[0, 0], 1.1048509233706119, places=PRECISION)

    def test_2d_gaussian_small_sigma_diff(self):
        tensor_train = np.array([[[5, 52]]], dtype=float)
        tensor_predict = np.array([[[5, 52, 1e-6, 1e-6, 0]]], dtype=float)
        loss = bivariate_gaussian_loss(tensor_train, tensor_predict).eval()
        self.assertAlmostEqual(loss[0, 0], 1.1048523660629765, places=PRECISION)

    def test_2d_gaussian_mu_ones(self):
        target = np.array([[[1, 1]]], dtype=float)
        prediction = np.array([[[1, 1, 1, 1, 0]]], dtype=float)
        loss = bivariate_gaussian_loss(target, prediction).eval()
        self.assertAlmostEqual(loss[0, 0], 2.3829037437816121, places=PRECISION)

    def test_2d_gaussian_mu_minus_ones(self):
        target = np.array([[[1, 1]]], dtype=float)
        prediction = np.array([[[1, 1, -1, -1, 0]]], dtype=float)
        loss = bivariate_gaussian_loss(target, prediction).eval()
        self.assertAlmostEqual(loss[0, 0], 2.3829037437816121, places=PRECISION)

    def test_2d_gaussian_ones(self):
        target = np.array([[[1, 1]]], dtype=float)
        prediction = np.array([[[1, 1, 1, 1, 1]]], dtype=float)
        loss = bivariate_gaussian_loss(target, prediction).eval()
        self.assertAlmostEqual(loss[0, 0], 1.9491232946784192, places=PRECISION)

    def test_2d_gaussian_rho_one(self):
        target = np.array([[[1, 2]]], dtype=float)
        prediction = np.array([[[1, 2, 0, 0, 1]]], dtype=float)
        loss = bivariate_gaussian_loss(target, prediction).eval()
        self.assertAlmostEqual(loss[0, 0], 1.9491232946784192, places=PRECISION)

    def test_2d_gaussian_rho_minus_one(self):
        target = np.array([[[1, 2]]], dtype=float)
        prediction = np.array([[[1, 2, 0, 0, -1]]], dtype=float)
        loss = bivariate_gaussian_loss(target, prediction).eval()
        self.assertAlmostEqual(loss[0, 0], 1.9491232946784192, places=PRECISION)

    def test_2d_gaussian_rho_two(self):
        target = np.array([[[1, 2]]], dtype=float)
        prediction = np.array([[[1, 2, 0, 0, 2]]], dtype=float)
        loss = bivariate_gaussian_loss(target, prediction).eval()
        self.assertAlmostEqual(loss[0, 0], 1.9491232946784192, places=PRECISION)

    def test_2d_gaussian_rho_minus_two(self):
        target = np.array([[[1, 2]]], dtype=float)
        prediction = np.array([[[1, 2, 0, 0, -2]]], dtype=float)
        loss = bivariate_gaussian_loss(target, prediction).eval()
        self.assertAlmostEqual(loss[0, 0], 1.9491232946784192, places=PRECISION)

    def test_2d_gaussian_big_diff(self):
        target = np.array([[[5, 52]]], dtype=float)
        prediction = np.array([[[1, 2, 3, 4, 5]]], dtype=float)
        loss = bivariate_gaussian_loss(target, prediction).eval()
        self.assertAlmostEqual(loss[0, 0], 16.11809565095832, places=PRECISION)

    def test_2d_gaussian_really_big_diff(self):
        target = np.array([[[5, 52]]], dtype=float)
        prediction = np.array([[[0, 0, 3, 4, 5]]], dtype=float)
        loss = bivariate_gaussian_loss(target, prediction).eval()
        self.assertAlmostEqual(loss[0, 0], 16.11809565095832, places=PRECISION)

    def test_2d_gaussian_max_neg_rho(self):
        min_rho = -19.06  # This is about the limit of rho before geom_gaussian_loss returns NaN
        target = np.array([[[5, 52]]], dtype=float)
        prediction = np.array([[[5, 52, -1, -1, min_rho]]], dtype=float)
        loss = bivariate_gaussian_loss(target, prediction).eval()
        self.assertAlmostEqual(loss[0, 0], -18.505382378927028, places=PRECISION)

    def test_2d_guassian_relative_loss(self):
        tensor0 = np.array([[[1, 2]]], dtype=float)
        tensor1 = np.array([[[1, 2, 0, 0, 0]]], dtype=float)
        tensor2 = np.array([[[1, 2, 1, 1, 0]]], dtype=float)
        tensor3 = np.array([[[1, 2, 1, 1, 1]]], dtype=float)
        tensor4 = np.array([[[1, 2, -1, -1, -1]]], dtype=float)
        tensor5 = np.array([[[1, 2, -1, -1, 4]]], dtype=float)
        loss1 = bivariate_gaussian_loss(tensor0, tensor1).eval()
        loss2 = bivariate_gaussian_loss(tensor0, tensor2).eval()
        loss3 = bivariate_gaussian_loss(tensor2, tensor3).eval()
        loss4 = bivariate_gaussian_loss(tensor0, tensor4).eval()
        loss4 = bivariate_gaussian_loss(tensor0, tensor4).eval()
        self.assertLess(loss1, loss2)
        self.assertLess(loss2, loss3)
        self.assertEqual(loss2, loss3)
        self.assertLess(loss1, loss3)


class TestGeomGaussianLoss(unittest.TestCase):
    def test_zero_loss(self):
        tensor1 = np.array([[[5, 52, 0, 0, 0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]]], dtype=float)
        tensor2 = np.array([[[5, 52, 0, 0, 0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]]], dtype=float)
        loss = geom_gaussian_loss(tensor1, tensor2).eval()

        self.assertAlmostEqual(loss[0, 0], 3.7079037516725526, places=PRECISION)

    def test_small_mu_loss(self):
        tensor1 = np.array([[[5, 52, 0, 0, 0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]]], dtype=float)
        tensor2 = np.array([[[5.01, 52.01, 0, 0, 0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]]], dtype=float)
        loss = geom_gaussian_loss(tensor1, tensor2).eval()

        self.assertAlmostEqual(loss[0, 0], 3.7080037516097173, places=PRECISION)

    def test_geom_type_loss(self):
        tensor1 = np.array([[[1, 2, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]], dtype=float)
        tensor2 = np.array([[[1, 2, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]]], dtype=float)
        loss = geom_gaussian_loss(tensor1, tensor2).eval()
        self.assertEqual(loss[0, 0], 6.6738155045999878)

    def test_geom_type_and_render_op_loss(self):
        tensor1 = np.array([[[1, 2, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]]], dtype=float)
        tensor2 = np.array([[[1, 2, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1]]], dtype=float)
        loss = geom_gaussian_loss(tensor1, tensor2).eval()
        self.assertEqual(loss[0, 0], 6.9147063722468438)

    def test_geom_one_coordinate_loss(self):
        tensor1 = np.array([[[1, 2, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]], dtype=float)
        tensor2 = np.array([[[1, 3, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]], dtype=float)
        loss = geom_gaussian_loss(tensor1, tensor2).eval()
        self.assertAlmostEqual(loss[0, 0], 6.7432698072295878, places=PRECISION)

    def test_geom_big_coordinate_loss(self):
        tensor1 = np.array([[[1, 2, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]], dtype=float)
        tensor2 = np.array([[[5, 52, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]], dtype=float)
        loss = geom_gaussian_loss(tensor1, tensor2).eval()
        self.assertAlmostEqual(loss[0, 0], 19.296149, places=PRECISION)

    def test_geom_big_inverted_coordinate_loss(self):
        tensor1 = np.array([[[5, 52, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]], dtype=float)
        tensor2 = np.array([[[-0.000369381, -0.000945276, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]], dtype=float)
        loss = geom_gaussian_loss(tensor1, tensor2).eval()
        self.assertAlmostEqual(loss[0, 0], 19.296149481306266, places=PRECISION)

    def test_geom_sigma_and_rho_loss(self):
        tensor1 = np.array([[[5, 52, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]], dtype=float)
        tensor2 = np.array([[[5, 52, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]], dtype=float)
        loss = geom_gaussian_loss(tensor1, tensor2).eval()
        self.assertAlmostEqual(loss[0, 0], 6.582147, places=PRECISION)

    def test_rho_diff_limit(self):
        min_rho = -19.06  # This is about the limit of rho before geom_gaussian_loss returns NaN
        tensor1 = np.array([[[5, 52, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]], dtype=float)
        tensor2 = np.array([[[5, 52, 0, 0, min_rho, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]], dtype=float)
        loss = geom_gaussian_loss(tensor1, tensor2).eval()
        self.assertAlmostEqual(loss[0, 0], -13.005895797801298, places=PRECISION)


class TestGaussian1d(unittest.TestCase):
    def test_zero_loss(self):
        target = np.array([[[1., 0.]]])
        prediction = np.array([[[1., 0.]]])
        loss = univariate_gaussian_loss(target, prediction).eval()
        self.assertAlmostEqual(loss[0, 0, 0], 0.73568186422336956, places=PRECISION)

    def test_one_mu_loss(self):
        target = np.array([[[1., 0.]]])
        prediction = np.array([[[0., 0.]]])
        loss = univariate_gaussian_loss(target, prediction).eval()
        self.assertAlmostEqual(loss[0, 0, 0], 0.90219415240769696, places=PRECISION)

    def test_minus_one_sigma_loss(self):
        target = np.array([[[1., 0.]]])
        prediction = np.array([[[1., -1.]]])
        loss = univariate_gaussian_loss(target, prediction).eval()
        self.assertAlmostEqual(loss[0, 0, 0], 1.0551951862023454, places=PRECISION)

    def test_minus_two_sigma_loss(self):
        target = np.array([[1., 0.]])
        prediction = np.array([[1., -2.]])
        loss = univariate_gaussian_loss(target, prediction).eval()
        self.assertAlmostEqual(loss[0, 0], 1.6143488206244256, places=PRECISION)

    def test_big_mu_sigma_diff(self):
        target = np.array([[52., 0.]])
        prediction = np.array([[0., 80]])  # Increase second value with .001 and the loss will jump to inf!
        loss = univariate_gaussian_loss(target, prediction).eval()
        self.assertAlmostEqual(loss[0, 0], 16.11809565095832, places=PRECISION)
