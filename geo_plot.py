
import json as js
import geopandas as gp
def loadMap(jsonMap):
    KommuneMap=js.load(jsonMap)
    KommuneCoordinates=pd.DataFrame(gp.GeoDataFrame.from_features(KommuneMap["features"]));
    KommuneCoordinates=KommuneCoordinates.drop_duplicates(subset='Kommunenummer', keep="first")
    Map=gp.GeoDataFrame(KommuneCoordinates);
    return Map
def orderData(reference, df):
    reference=pd.DataFrame(reference)
    R=reference.merge(df,'inner')
    return R
def plotMap(data, ref, axis, **kwargs):
    A=orderData(ref,data)
    gp.GeoDataFrame(A).plot(A.columns[-1], cmap=cm.get_cmap('seismic'),ax=axis, 
                            legend=True )
    plt.gca().axes.get_xaxis().set_visible(False);
    plt.yticks([])
    plt.xticks([])
    return
