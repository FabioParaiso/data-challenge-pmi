import seaborn as sns
import matplotlib.pyplot as plt
import folium
import h3


def mobility_map(cab_data):
    plt.figure(figsize=(10, 10))
    sns.scatterplot(data=cab_data, x="longitude", y="latitude", alpha=0.1)


def feature_relations(cab_data):
    sns.pairplot(data=cab_data, kind='scatter', diag_kind='kde')


def feature_over_time(cab_data, y_variable):
    plt.figure(figsize=(25, 8))
    sns.lineplot(data=cab_data, x='time', y=y_variable, hue='cab_id')


def visualize_hexagons(hexagons, color="red", folium_map=None):
    """
    hexagons is a list of hexcluster. Each hexcluster is a list of hexagons.
    eg. [[hex1, hex2], [hex3, hex4]]
    """
    polylines = []
    lat = []
    lng = []
    for hex in hexagons:
        polygons = h3.h3_set_to_multi_polygon([hex], geo_json=False)
        # flatten polygons into loops.
        outlines = [loop for polygon in polygons for loop in polygon]
        polyline = [outline + [outline[0]] for outline in outlines][0]
        lat.extend(map(lambda v: v[0], polyline))
        lng.extend(map(lambda v: v[1], polyline))
        polylines.append(polyline)

    if folium_map is None:
        m = folium.Map(location=[sum(lat) / len(lat), sum(lng) / len(lng)], zoom_start=13, tiles='cartodbpositron')
    else:
        m = folium_map
    for polyline in polylines:
        my_PolyLine = folium.PolyLine(locations=polyline, weight=8, color=color)
        m.add_child(my_PolyLine)
    return m


def visualize_polygon(polyline, color):
    polyline.append(polyline[0])
    lat = [p[0] for p in polyline]
    lng = [p[1] for p in polyline]
    m = folium.Map(location=[sum(lat) / len(lat), sum(lng) / len(lng)], zoom_start=13, tiles='cartodbpositron')
    my_PolyLine = folium.PolyLine(locations=polyline, weight=8, color=color)
    m.add_child(my_PolyLine)
    return m


def visualize_time_series(df):
    plt.figure(figsize=(25, 8))
    df.plot()
