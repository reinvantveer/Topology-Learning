import unittest
import pandas
import numpy as np

from GeoVectorizer import GeoVectorizer, GEO_VECTOR_LEN, RENDER_INDEX, FULL_STOP_INDEX
from test_files import gmm_output

TOPOLOGY_CSV = 'test_files/polygon_multipolygon.csv'
SOURCE_DATA = pandas.read_csv(TOPOLOGY_CSV)
brt_wkt = SOURCE_DATA['brt_wkt']
osm_wkt = SOURCE_DATA['osm_wkt']
target_wkt = SOURCE_DATA['intersection_wkt']

input_geom = np.array([
    [0, 0, 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0.],
    [0, 1, 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0.],
    [1, 1, 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0.],
    [1, 0, 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0.],
    [0, 0, 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0.],
    [0, 0, 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., ],
    [0, -1, 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., ],
    [-1, -1, 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., ],
    [-1, 0, 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., ],
    [0, 0, 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1., ],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., ]
])

output_geom = np.array([
    [0, 0, 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0.],
    [0, 0.25, 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0.],
    [0, 0.5, 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0.],
    [0, 0.75, 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0.],
    [0, 1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0.],
    [0.25, 1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0.],
    [0.5, 1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0.],
    [1., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0.],
    [1., 0.5, 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0.],
    [1., 0, 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0.],
    [0.5, 0, 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0.],
    [0, 0, 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0.],
    [0, 0, 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., ],
    [0, -0.5, 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., ],
    [0, -1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., ],
    [-0.5, -1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., ],
    [-1., -1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., ],
    [-1., -0.5, 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., ],
    [-1., 0, 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., ],
    [-0.5, 0, 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., ],
    [0, 0, 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1., ],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., ]
])


class TestVectorizer(unittest.TestCase):
    def test_max_points(self):
        max_points = GeoVectorizer.max_points(brt_wkt, osm_wkt)
        self.assertEqual(max_points, 159)

    def test_interpolate(self):
        interpolated = GeoVectorizer.interpolate(input_geom, len(input_geom) * 2)
        for index, _ in enumerate(interpolated):
            result = list(interpolated[index])
            expected = list(output_geom[index])
            self.assertListEqual(result, expected, msg='Lists differ at index %i' % index)

    def test_vectorize_one_wkt(self):
        max_points = GeoVectorizer.max_points(brt_wkt, osm_wkt)
        vectorized = []
        target_set = SOURCE_DATA['intersection_wkt']
        for index in range(len(target_set)):
            vectorized.append(GeoVectorizer.vectorize_wkt(target_set[index], max_points))
        self.assertEqual(len(target_set), len(brt_wkt))

    def test_vectorize_big_multipolygon(self):
        with open('test_files/big_multipolygon_wkt.txt', 'r') as file:
            wkt = file.read()
            max_points = GeoVectorizer.max_points([wkt])
            vectorized = GeoVectorizer.vectorize_wkt(wkt, max_points)
            self.assertEqual(len(vectorized), max_points)

    def test_vectorize_polygon_gt_max_points_error(self):
        with open('test_files/big_multipolygon_wkt.txt', 'r') as file:
            wkt = file.read()
            max_points = 50
            with self.assertRaises(ValueError) as ve:
                GeoVectorizer.vectorize_wkt(wkt, max_points)
            self.assertEqual("The number of points in the geometry exceeds", str(ve.exception)[0:44])

    def test_simplify_polygon_gt_max_points(self):
        with open('test_files/big_multipolygon_wkt.txt', 'r') as file:
            wkt = file.read()
            max_points = 70
            vectorized = GeoVectorizer.vectorize_wkt(wkt, max_points, simplify=True)
            self.assertEqual(len(vectorized), max_points)

    def test_simplify_multipolygon_gt_max_points(self):
        with open('test_files/multipart_multipolygon_wkt.txt', 'r') as file:
            wkt = file.read()
            max_points = 20
            vectorized = GeoVectorizer.vectorize_wkt(wkt, max_points, simplify=True)
            self.assertEqual(vectorized.shape, (max_points, GEO_VECTOR_LEN))

    def test_wkt_vectorize_two_wkt(self):
        vectorized = []
        max_points = GeoVectorizer.max_points(brt_wkt, osm_wkt)
        for index in range(len(brt_wkt)):
            vectorized.append(GeoVectorizer.vectorize_two_wkts(brt_wkt[index], osm_wkt[index], max_points))

        num_records = len(vectorized)
        num_points = len(vectorized[0])
        num_features = len(vectorized[0][0])

        self.assertEqual(num_records, 13)
        self.assertEqual(num_points, 159)
        self.assertEqual(num_features, GEO_VECTOR_LEN)

        for record in vectorized:
            points = [point for point in record if point[0] > 0]
            # Every first point should have a "render" code
            self.assertEqual(points[0][RENDER_INDEX], 1)
            # Every last point should have a "full stop" code
            self.assertEqual(points[-1][FULL_STOP_INDEX], 1)

    def test_decypher(self):
        self.maxDiff = None
        max_points = GeoVectorizer.max_points(brt_wkt, osm_wkt)
        target_vector = GeoVectorizer.vectorize_wkt(target_wkt[0], max_points)
        decyphered = GeoVectorizer.decypher(target_vector)
        self.assertEqual(decyphered, target_wkt[0])

    def test_decypher_prediction(self):
        self.maxDiff = None
        max_points = GeoVectorizer.max_points(brt_wkt, osm_wkt)
        target_vector = GeoVectorizer.vectorize_wkt(target_wkt[0], max_points)
        decyphered = GeoVectorizer.decypher(target_vector)
        self.assertEqual(decyphered, target_wkt[0])

    def test_decypher_gmm_sample(self):
        pred = gmm_output.prediction
        sample_size = 10
        points = GeoVectorizer(gmm_size=5).decypher_gmm_geom(pred, sample_size=sample_size)
        self.assertEqual(len(points), 40)
        self.assertEqual(points.geom_type, "MultiPoint")

    def test_decypher_gmm_geom(self):
        pred = gmm_output.prediction
        geom = GeoVectorizer(gmm_size=5).decypher_gmm_geom(pred)
        self.assertEqual(geom.wkt, "POLYGON ((0.000192008913 -0.00157661038, -0.000827540178 -0.00121512916, "
                                   "-0.000584304798 -0.00181529298, 0.00231453031 -0.00123368669, 0.000192008913 "
                                   "-0.00157661038))")
