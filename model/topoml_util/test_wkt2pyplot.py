import unittest
from datetime import datetime
from wkt2pyplot import wkt2pyplot


class TestWktToPyplotPoly(unittest.TestCase):
    def test_geometrycollection_empty(self):
        inputs = ['GEOMETRYCOLLECTION EMPTY']  # This is valid WKT
        plt, fig, ax = wkt2pyplot(inputs)
        plt.show()  # It should show an empty plot

    def test_polygon_conversion(self):
        TIMESTAMP = str(datetime.now()).replace(':', '.')

        inputs = 'POLYGON((1.09872727273 -0.289454545452,-0.241272727273 0.682545454538,-0.992272727274 ' \
                 '0.292545454528,0.347727272727 -0.680454545474,1.09872727273 -0.289454545452))\nPOLYGON((' \
                 '-0.976272727273 0.302545454574,-0.25627272727 0.676545454539,1.05372727273 -0.276454545443,' \
                 '0.320727272731 -0.654454545455,-0.477272727268 -0.0664545454754,-0.976272727273 ' \
                 '0.302545454574))'
        inputs = inputs.split('\n')

        target = 'POLYGON((-0.974272727277 0.301545454562,-0.255272727276 0.675545454527,1.05372727273 ' \
                 '-0.276454545443,0.320727272731 -0.654454545455,-0.477272727268 -0.0664545454754,-0.974272727277 ' \
                 '0.301545454562))'

        prediction = [
            'POINT(-0.974272727277 0.301545454562)',
            'POINT(-0.255272727276 0.675545454527)',
            'POINT(1.05372727273 -0.276454545443)',
            'POINT(0.320727272731 -0.654454545455)',
            'POINT(-0.477272727268 -0.0664545454754)',
            'POINT(-0.974272727277 0.301545454562)',
        ]
        plt, fig, ax = wkt2pyplot(inputs, [target], prediction)
        plt.text(0.01, 0.06, 'prediction: some more text', transform=ax.transAxes)
        plt.text(0.01, 0.01, 'target: some text', transform=ax.transAxes)

        plt.show()
