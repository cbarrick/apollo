import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import scipy.interpolate

from collections import namedtuple
from mpl_toolkits.basemap import Basemap

# A mapping from city codes to (name, lat, lon)
CITIES = {
    'ALAPAHA': ('Alapaha', 31.34471, -83.24072),
    'ALBANY': ('Albany', 31.55402, -84.05194),
    'ALMA': ('Alma', 31.54286, -82.51002),
    'ALPHARET': ('Alpharetta', 34.09204, -84.22242),
    'ARABI': ('Arabi', 31.82782, -83.81641),
    'ARLINGT': ('Arlington', 31.35322, -84.63083),
    'ATLANTA': ('Atlanta', 33.74789, -84.41439),
    'ATTAPUL': ('Attapulgus', 30.76156, -84.48529),
    'BAXLEY': ('Baxley', 31.75127, -82.44431),
    'BLAIRSVI': ('Blairsville', 34.83897, -83.92807),
    'BLURIDGE': ('Blue Ridge', 34.88512, -84.34240),
    'BOWEN': ('Tifton-Bowen', 31.48076, -83.43913),
    'BRUNSW': ('Brunswick', 31.18550, -81.48283),
    'BYROMVIL': ('Byromville', 32.18962, -83.87672),
    'BYRON': ('Byron', 32.65755, -83.73745),
    'CAIRO': ('Cairo', 30.86181, -84.23141),
    'CALHOUN': ('Calhoun', 34.55761, -84.81579),
    'CAMILLA': ('Camilla', 31.28008, -84.29161),
    'CLARKSHI': ('Clarks Hill-SC', 33.65989, -82.19444),
    'CORDELE': ('Cordele', 32.02339, -83.94076),
    'COVING': ('Covington', 33.43850, -83.84468),
    'DAHLON': ('Dahlonega', 34.61074, -83.87932),
    'DALLAS': ('Dallas', 33.89322, -84.83533),
    'DANVILLE': ('Danielsville', 34.14541, -83.23844),
    'DAWSON': ('Dawson', 31.75824, -84.43584),
    'DEARING': ('Dearing', 33.36501, -82.40302),
    'DIXIE': ('Dixie', 30.79448, -83.66747),
    'DONALSON': ('Donalsonville', 31.01538, -84.87981),
    'DOUGLAS': ('Douglas', 31.48864, -82.78500),
    'DUBLIN': ('Dublin', 32.49722, -82.92290),
    'DUCKER': ('Ducker', 31.53322, -84.36933),
    'DUNWOODY': ('Dunwoody', 33.98924, -84.36160),
    'EATONTON': ('Eatonton', 33.39693, -83.48822),
    'ELBERTON': ('Elberton', 34.02028, -82.60607),
    'ELLIJAY': ('Ellijay', 34.61980, -84.37405),
    'FLOYD': ('Rome', 34.34328, -85.11632),
    'FTVALLEY': ('Fort Valley', 32.53069, -83.89024),
    'GAINES': ('Gainesville', 34.35205, -83.79279),
    'GEORGETO': ('Georgetown', 31.77596, -85.06052),
    'GRIFFIN': ('Griffin', 33.26291, -84.28384),
    'HATLEY': ('Hatley', 31.92156, -83.62568),
    'HOMERV': ('Homerville', 31.02077, -82.66101),
    'JVILLE': ('Jeffersonville', 32.68256, -83.35946),
    'JONESB': ('Jonesboro', 33.52180, -84.31390),
    'LAFAYET': ('LaFayette', 34.75989, -85.38421),
    'MCRAE': ('McRae', 32.07700, -82.90200),
    'MIDVILLE': ('Midville', 32.87560, -82.21609),
    'MOULTRIE': ('Moultrie', 31.14589, -83.71642),
    'NAHUNTA': ('Nahunta', 31.17990, -82.00861),
    'NEWTON': ('Newton', 31.22391, -84.47790),
    'OAKWOOD': ('Oakwood', 34.23700, -83.87102),
    'ODUM': ('Odum', 31.63292, -82.05171),
    'OSSABAW': ('Ossabaw', 31.83784, -81.09116),
    'PLAINS': ('Plains', 32.04676, -84.37102),
    'SANLUIS': ('San Luis-CR', 10.28290, -84.79857),
    'SASSER': ('Sasser', 31.70896, -84.36794),
    'SAVANNAH': ('Savannah', 31.99807, -81.26878),
    'SHELLMAN': ('Shellman', 31.74470, -84.61144),
    'SKIDAWAY': ('Skidaway', 31.94556, -81.03506),
    'SPARTA': ('Sparta', 33.27780, -82.96622),
    'STATES': ('Statesboro', 32.48523, -81.81386),
    'TENNILLE': ('Tennille', 32.92698, -82.79782),
    'TIFTON': ('Tifton', 31.49416, -83.52634),
    'TIGER': ('Tiger', 34.84738, -83.42718),
    'TYTY': ('TyTy', 31.50911, -83.64813),
    'UNADILLA': ('Unadilla', 32.25894, -83.66154),
    'VALDOSTA': ('Valdosta', 30.82442, -83.31546),
    'VIDALIA': ('Vidalia', 32.14000, -82.34511),
    'VIENNA': ('Vienna', 32.11120, -83.67544),
    'WANSLEY': ('Roopville', 33.42236, -85.05521),
    'WATHORT': ('Watkinsville-Hort', 33.88689, -83.41941),
    'WATUGA': ('Watkinsville-UGA', 33.87285, -83.53545),
    'WATUSDA': ('Watkinsville-USDA', 33.86930, -83.44993),
    'WOODBINE': ('Woodbine', 30.96235, -81.77383),
}


class GeorgiaMap(Basemap):
    width = 530e3
    height = 545e3
    center = (32.8, -83.6)  # (lat, lon) of Macon, GA
    bottom = 30.0
    top = 36.0

    def __init__(self, data, resolution='l'):
        Basemap.__init__(
            self,
            width=GeorgiaMap.width,
            height=GeorgiaMap.height,
            projection='lcc',
            lat_0=GeorgiaMap.center[0],
            lon_0=GeorgiaMap.center[1],
            lat_1=GeorgiaMap.top,
            lat_2=GeorgiaMap.bottom,
            resolution=resolution)

        self.data = dict(data)

    def draw(self, nx=100, ny=100):
        n = len(self.data)
        data, lons, lats = self.interpolate(nx, ny)
        cs = self.pcolormesh(lons, lats, data, latlon=True)
        self.drawcoastlines()
        self.drawcountries()
        self.drawstates()
        self.drawmapboundary()
        self.colorbar(cs)
        for (city, data) in self.data.items():
            s = '{:.2f}'.format(np.mean(data))
            lon = CITIES[city][2]
            lat = CITIES[city][1]
            self.text(lon, lat, s, latlon=True)

    def plot(self, city, *args, **kwargs):
        city = CITIES[city.upper()]
        lon = city[2]
        lat = city[1]
        x, y = self(lon, lat)
        Basemap.plot(self, x, y, *args, **kwargs)

    def interpolate(self, nx, ny):
        lons = [CITIES[c][2] for c in self.data.keys()]
        lats = [CITIES[c][1] for c in self.data.keys()]
        data = [np.mean(l) for l in self.data.values()]
        interp = sp.interpolate.Rbf(lons, lats, data)
        lons, lats = self.makegrid(nx, ny)
        data = np.ndarray((nx, ny))
        for i in range(nx):
            for j in range(ny):
                data[i][j] = interp(lons[i][j], lats[i][j])
        return data, lons, lats

    def text(self, x, y, s, fontdict=None, withdash=False, **kwargs):
        latlon = kwargs.pop('latlon', False)
        if latlon: x, y = self(x, y)
        return plt.text(x, y, s, fontdict, withdash, **kwargs)
