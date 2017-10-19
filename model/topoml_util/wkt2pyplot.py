from datetime import datetime

import matplotlib
import os

if not os.environ.get('MATPLOTLIB_TEST'):
    matplotlib.use('Agg')  # for headless machine instances

from shapely import wkt
from matplotlib import pyplot as plt


def wkt2pyplot(input_wkts, target_wkts=None, prediction_wkts=None,
               input_color='green', target_color='red', pred_color='blue'):
    """
    Convert arrays of input, target and prediction well-known encoded geometry arrays to pyplot
    :param input_wkts: an array of input geometries, rendered in (standard) green
    :param target_wkts: optional array of target geometries, rendered in (standard) red
    :param prediction_wkts: optional array of prediction geometries, rendered in (standard) blue
    :param input_color: a pyplot-compatible notation of color, default green
    :param pred_color: a pyplot-compatible notation of color, default red
    :param target_color: a pyplot-compatible notation of color, default blue
    :return: a matplotlib pyplot fig, ax and plt
    """
    input_geoms = [wkt.loads(input_wkt) for input_wkt in input_wkts]

    fig, ax = plt.subplots()

    input_polys = []
    for input_geom in input_geoms:
        if len(input_geom.bounds) > 0 and input_geom.geom_type == 'Polygon':
            input_polys.append(matplotlib.patches.Polygon(input_geom.boundary.coords))

    inputs = matplotlib.collections.PatchCollection(input_polys, alpha=0.4, linewidth=1)
    inputs.set_color(input_color)
    ax.add_collection(inputs)

    # target_polys = [Polygon(target_geom.boundary.coords) for target_geom in target_geoms]
    # targets = PatchCollection(target_polys, alpha=0.4, linewidth=1)
    # targets.set_color(target_color)
    # ax.add_collection(targets)

    # TODO: handle other types of geometries
    # TODO: handle holes in polygons (donuts)
    if target_wkts:
        target_geoms = [wkt.loads(target_wkt) for target_wkt in target_wkts]
        for geom in target_geoms:
            if geom.type == 'Point':
                plt.plot(geom.coords.xy[0][0], geom.coords.xy[1][0],
                         marker='o', color=target_color, alpha=0.4, linewidth=0)
            elif geom.type == 'Polygon':
                collection = matplotlib.collections.PatchCollection([matplotlib.patches.Polygon(geom.boundary.coords)],
                                                                    alpha=0.4, linewidth=1)
                collection.set_color(target_color)
                ax.add_collection(collection)

    if prediction_wkts:
        prediction_geoms = [wkt.loads(prediction_wkt) for prediction_wkt in prediction_wkts]
        for geom in prediction_geoms:
            if geom.geom_type == 'Point':
                plt.plot(geom.coords.xy[0][0], geom.coords.xy[1][0],
                         marker='o', color=pred_color, alpha=0.1, linewidth=0)
            elif geom.type == 'Polygon':
                collection = matplotlib.collections.PatchCollection([matplotlib.patches.Polygon(geom.boundary.coords)],
                                                                    alpha=0.4, linewidth=1)
                collection.set_color(pred_color)
                ax.add_collection(collection)

    plt.axis('auto')

    return plt, fig, ax


def save_plot(geoms, plot_dir='plots', timestamp=None):
    plt, fig, ax = wkt2pyplot(*geoms)
    plt.savefig(plot_dir + '/plt_' + timestamp + '.png')
    plt.close('all')
