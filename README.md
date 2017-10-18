# Topology Learning
A machine learning project for learning geospatial topology

# Data preparation
Note first that there are pre-built numpy archive files `geodata-vectorized.npz` under `files`, so you don't need to rebuild the training data. If you want to prepare the data yourself, or want to derive a different pipeline from it, I suggest you go for the dockerized version (if you value your time). The dockerized version uses a PostGIS database instance to implement an ETL process that does the heavy lifting. Afterwards it's mostly a question of converting to normalized numpy vectors that can be understood by machine learning frameworks and saving the data to numpy archives.

## Numpy archive description
The numpy archive `geodata-vectorized.npz` under `files` contains vectors deserialized from well-known text geometries, using the [shapely](https://pypi.python.org/pypi/Shapely) library. They are re-serialized as a 3D tensor as a combination of real-valued and one-hot components:

|Script|Description|
|:------|:----|
|input_geoms| A nested array of shape (?, ?, 16) - depending on the sequence length settings - of deserialized WKT polygons with the following encoding: `[:, :, 0:5]`:  The polygon point/node coordinates in lon/lat, `[:, :, 5:13]`: The geometry type, a one-hot encoded sequence of set {GeometryCollection, Point, LineString, Polygon, MultiPoint, MultiLineString, MultiPolygon, Geometry}, `[:, :, 13:]`:  A one-hot encoded sequence of actions in set {render, stop, full stop}
|intersection|Geometries representing the intersection in WGS84 lon/lat, same geometry encoding as `input_geoms`
|centroid_distance|Distance as gaussian with parameters mean and sigma shape `[:, :, 2]` in meters between the centroids
|geom_distance|Distance as gaussian with mean and sigma shape `[:, :, 2]` in meters between the geometries, distance 0 if intersecting
|brt_centroid|Centroid point in WGS84 lon/lat of the BRT geometry
|osm_centroid|Centroid point in WGS84 lon/lat of the OSM geometry
|centroids|Two centroid points for BRT and OSM in WGS84 lon/lat
|centroids_rd|Two centroid points for BRT and OSM in Netherlands RD meters

## Docker
Due to the cross-platform dependency issues installing python geospatial packages, the setup has been [Dockerized](https://www.docker.com/) to ensure cross-platform interoperability. This is the easiest way to use. 

- See [instructions](https://docs.docker.com/engine/installation/#supported-platforms) on installing and running Docker on your platform.
- Install [docker compose](https://docs.docker.com/compose/install/)
- Run `docker-compose build` in the root folder of this repository
- **!! Make sure `./script/get-data.sh` has LF line endings if you run this on windows!!**. To be sure, you can install [Git for windows](https://git-for-windows.github.io/), open a git bash shell in the repo folder and issue a `dos2unix ./script/*.sh`. This will convert the line endings to unix line feeds.
- Run `docker-compose up`. Once the container has exited, you should see a `files` directory in your repo folder with a file `topology-training.csv`. This contains your extracted geometries, distances and intersections. 
- You can re-create the topology-training csv using `docker-compose run data-prep export-data.sh`

## Using self-installed libraries
If you need to run or debug locally, you need the following dependencies:

|Package|Linux|Mac|Windows|
|:------|:----|:---|:------|
|Shapely|`apt-get install -y libgeos-dev && pip install shapely`| |download and `pip install` [one of these wheels](http://www.lfd.uci.edu/~gohlke/pythonlibs/#shapely)|
|GDAL|`apt-get install -y libproj-dev && git clone https://github.com/OSGeo/gdal.git && cd gdal/gdal && ./configure --with-python && make >/dev/null 2>&1 && make install && ldconfig`| |download and `pip install` [one of these wheels](http://www.lfd.uci.edu/~gohlke/pythonlibs/#gdal)|
|PostGIS|`apt-get install postgres postgis` | `brew install -y postgres postgis`| see [here](http://postgis.net/windows_downloads/) |
|slackclient|`pip3 install slackclient`|`pip3 install slackclient`|`pip3 install slackclient`|
Once you have everything installed, cd to the prep directory and execute `get-data.sh` en `export-data.sh`.

## Conversion to numpy archive
After you have created your `topology-training.csv`, convert it to a numpy archive using `cd model && python3 vectorize.py`. Note there is a pre-built `geodata-vectorized.npz` under `files`

# Running the models
Model scripts are run from the `model` directory. Each one of them is a more or less elaborate strategy to work with real-valued data, often combined with categorical data, using LSTMs.

You can monitor progress using tensorboard:
`tensorboard --log_dir ./model/tensorboard`
or
`python3 -m tensorflow.tensorboard --logdir ./model/tensorboard`

## Gaussian verification experiments
Many models include gaussian parameter outputs as part of the task. The job of these models is to verify the accuracy and stability of univariate and bivariate gaussian loss functions.

### `fixed_univariate_gaussian.py`
The most simple of gaussian approximations. Attempts to approximate a mu of 52. as close as possible with a sigma of 0. Converges in about 10 epochs to exactly [52. 0.].


### `fixed_bivariate_gaussian.py`
Attempts to approximate a bivariate gaussian with mu's of a realistic coordinate set of [5, 52], sigmas of 0 and rho of 0. Converges in about 30 epochs, but weirdly brings rho towards +/- 9e0 instead of 0.

### `random_univariate_gaussian.py`
Trains to approximate a randomly instantiated set of integers between 1 and 20. 


## Distance models

### `centroid_distance.py`
Calculate the distance between the centroids (of two polygons) in meters, expressed as a single gaussian. Converges to deviations of centimeters in about 50 epochs. This is a helpful demo model, since centroidal distances can in general practice be used between any set of two (multi)geometries, whether this are (multi)points, (multi)polygons or other.

### `geom_distance.py`
Approximates the distance between two polygons in meters, expressed as a single gaussian. This is an approximation of a vanilla distance query such as [PostGIS:ST_distance](http://postgis.net/docs/ST_Distance.html). If the polygons intersect, the distance is 0. Converges in about 25 epochs to centimeter-level precision. Intersecting geometries do not converge to exactly 0, but to levels within centimeters deviation.


## Intersection sequence-to-sequence auto-encoder

### `intersection.py` 
Approximates the intersection of two geometries into a new geometry. Work in progress.

### `intersection-concat.py` 
Approximates the intersection of two concatenated geometries into a new geometry, a variation of the previous setup. Work in progress.

## Character-level model
There is one model that uses a character-level sequence-to-sequence strategy to approximate intersections. Work in progress.
