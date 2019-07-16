'''Additional vizualizations for matplotlib.
'''

import numpy as np
import pandas as pd

import cartopy
import cartopy.crs as ccrs
import cartopy.feature
import cartopy.mpl.geoaxes

import matplotlib as mpl
import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1 import make_axes_locatable

import apollo


DAYS = np.array(['Sun.', 'Mon.', 'Tues.', 'Wed.', 'Thurs.', 'Fri.', 'Sat.'])
MONTHS = np.array(['Jan.', 'Feb.', 'Mar.', 'Apr.', 'May', 'June',
        'July', 'Aug.', 'Sept.', 'Oct.', 'Nov.', 'Dec.'])


def date_heatmap(series, start=None, end=None, mean=False, edgecolor='black',
        ax=None, **kwargs):
    '''Plot a calendar heatmap given a datetime series.

    Arguments:
        series (pd.Series):
            A series of numeric values with a datetime index. The index is
            interpreted by :func:`pandas.to_datetime`. Values occurring on the
            same day are combined by sum.
        start (Any):
            The first day to be considered in the plot. The value is interpreted
            by :func:`pandas.to_datetime`. The default is the earliest date in
            the index.
        end (Any):
            The last day to be considered in the plot. The value is interpreted
            by :func:`pandas.to_datetime`. The default is the latest date in
            the index.
        mean (bool):
            Combine values occurring on the same day by mean instead of sum.
        edgecolor (str):
            The color of the edges that delineate months.
        ax (matplotlib.axes.Axes or None):
            The axes on which to draw the heatmap. The default is the current
            axes in the :mod:`~matplotlib.pyplot` API.
        **kwargs:
            Forwarded to :meth:`~matplotlib.axes.Axes.pcolormesh` for drawing
            the heatmap.

    Returns:
        matplotlib.axes.Axes:
            The axes on which the heatmap was drawn. It is set as the current
            axes in the :mod:`~matplotlib.pyplot` API.

    Example:
        .. figure:: /_static/heatmap.png
    '''
    # Interpret the index if it is not already a datetime index.
    index = pd.to_datetime(series.index)
    series = pd.Series(series.values, index)

    # Combine values occurring on the same day.
    dates = series.index.floor('D')
    group = series.groupby(dates)
    series = group.mean() if mean else group.sum()

    # Parse start/end, defaulting to the min/max of the index.
    start = pd.to_datetime(start or series.index.min())
    end = pd.to_datetime(end or series.index.max())

    # We use [start, end) as a half-open interval below.
    # This effects the Sunday formulas.
    end += np.timedelta64(1, 'D')

    # Get the previous/following Sunday to start/end.
    # Pandas and numpy day-of-week conventions are Monday=0 and Sunday=6.
    start_sun = start - np.timedelta64((start.dayofweek + 1) % 7, 'D')
    end_sun = end + np.timedelta64(7 - end.dayofweek - 1, 'D')

    # Compute the heatmap, month ticks, and month edges.
    # Days on the calendar but outside [start, end) are set to NaN.
    n_weeks = (end_sun - start_sun).days // 7
    heatmap = np.full((7, n_weeks), np.nan)
    ticks = {}
    edges_top = []
    edges_left = []
    for week in range(n_weeks):
        for day in range(7):
            date = start_sun + np.timedelta64(7 * week + day, 'D')
            if date.day == 1:
                edges_top.append((week, day))
            if date.day < 8:
                edges_left.append((week, day))
            if date.day == 15:
                ticks[week] = MONTHS[date.month - 1]
            if date.dayofyear == 15:
                ticks[week] += f'\n{date.year}'
            if start <= date < end:
                heatmap[day, week] = series.get(date, 0)

    # Offset the coordinates by 0.5 to align the month and week ticks.
    y = np.arange(8) - 0.5
    x = np.arange(n_weeks + 1) - 0.5

    # Plot the heatmap. We must Invert the y-axis for pcolormesh, but not for
    # imshow. Prefer pcolormesh to better support vector formats.
    ax = ax or plt.gca()
    ax.invert_yaxis()
    mesh = ax.pcolormesh(x, y, heatmap, **kwargs)

    # Draw lines delineating months.
    for x, y in edges_top:
        ax.hlines(y-0.5, x-0.5, x+0.5, color=edgecolor)
    for x, y in edges_left:
        ax.vlines(x-0.5, y-0.5, y+0.5, color=edgecolor)

    # Set the ticks.
    ax.set_xticks(list(ticks.keys()))
    ax.set_xticklabels(list(ticks.values()))
    ax.set_yticks([1, 3, 5])
    ax.set_yticklabels(DAYS[[1, 3, 5]])

    # Set the current image and axes in the pyplot API.
    plt.sca(ax)
    plt.sci(mesh)

    return ax


def date_heatmap_figure(series, cmap=None, bad_color=None, scale=1,
        savefig=None):
    '''Plot a calendar heatmap on a new figure.

    Arguments:
        series (pd.Series):
            A series of numeric values with a datetime index. The index is
            interpreted by :func:`pandas.to_datetime`. Values occurring on the
            same day are combined by sum.
        cmap (matplotlib.colors.Colormap or str or None):
            The colormap for the plot. The default is taken from the
            :mod:`~matplotlib.pyplot` API. If the series has an integer dtype,
            the cmap is discretized.
        bad_color (Sequence[float] or str or None):
            The color to use for out-of-bounds dates. The default is taken from
            the :mod:`~matplotlib.pyplot` API.
        scale (int or float):
            A scale factor for the plot. Note that the font size is fixed, so a
            larger scale results in smaller labels relative to the plot.
        savefig (str or None):
            If set, the figure is saved with the given name before returning.
            The reccomended value is ``'heatmap.pdf'``. Since PDF is a vector
            format it will often yield the best image and smallest file.

    Returns:
        matplotlib.figure.Figure:
            The figure that was drawn.

    Example:
        .. figure:: /_static/heatmap.png
    '''
    # Generate random data if none is given. Useful for testing.
    if series is None:
        index = apollo.date_range(start='2016-12-01', end='2019-05-28')
        series = np.random.randint(7, size=len(index))
        series = pd.Series(series, index)

    # Create a figure whose aspect ratio matches the heatmap.
    dates = series.index
    start, end = dates.min(), dates.max()
    n_days = (end - start).days
    n_weeks = n_days / 7
    w, h = plt.figaspect(7 / n_weeks)
    fig = plt.figure(figsize=(w * scale, h * scale))

    # Plot the heatmap. The 'equal' aspect makes all cells square, but makes
    # sizing the rest of the plot really annyoing.
    ax = date_heatmap(series)
    ax.set_aspect('equal')
    mesh = plt.gci()

    # Plot the color bar. We use the AxesGrid toolkit to ensure the color bar
    # is always the same height as the heatmap.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", 0.1, pad=0.1)
    plt.colorbar(cax=cax)

    # Set the colors. If the series has a signed or unsigned integer dtype, then
    # we get a new discretized the colormap based on the first. In that case, we
    # offset the color limits by 0.5 to center the ticks on the color bar.
    cmap = plt.get_cmap(cmap)
    if series.dtype.kind in 'iu':
        cmin, cmax = mesh.get_clim()
        mesh.set_clim(cmin - 0.5, cmax + 0.5)
        n_colors = int(cmax - cmin) + 1
        cmap = plt.get_cmap(cmap.name, n_colors)
    if bad_color is not None:
        cmap.set_bad(bad_color)
    mesh.set_cmap(cmap)

    # Save to a file. PDF is a vector format and will be smaller than a raster
    # image like PNG. PDFs can be used as images in LaTeX with `graphicx`.
    if savefig is not None:
        fig.savefig(savefig, bbox_inches='tight')

    return fig


MAP_FEATURES = {
    'coastlines': cartopy.feature.NaturalEarthFeature('physical', 'coastline', '110m'),
    'countries': cartopy.feature.NaturalEarthFeature('cultural', 'admin_0_countries', '110m'),
    'states': cartopy.feature.NaturalEarthFeature('cultural', 'admin_1_states_provinces', '110m'),
}


def nam_map(xrds, feature, reftime=0, forecast=0, level=0, title=None,
        detail='states', scale='10m', cmap='viridis', **kwargs):
    '''Plot NAM data as a map on a new figure.

    This function downloads shape files to draw the map. This may take a while
    the first time you use a specific combination of ``detail`` and ``scale``.

    Arguments:
        xrds (xarray.Dataset):
            The dataset of NAM data containing the feature to plot.
        feature (str):
            The name pf the feature being plotted.
        reftime (int or timestamp):
            The reference time of the data being plotted. If given as an
            integer, it is interpreted as an index along the reftime axis.
            Otherwise, it is interpreted as a :class:`pandas.Timestamp` naming
            the reftime.
        forecast (int):
            The forecast hour of the data being plotted.
        level (int):
            The index along the z-axis of the data to plot.
        title (str or None):
            The title of the figure. The default title combines the reftime and
            forecast hour.
        detail (str):
            The level of detail of the map. Recognized values from most to least
            detailed include ``'states'``, ``'countries'``, ``'coastlines'``.
        scale (str):
            The scale of the map details. The value ``'110m'`` means a scale of
            1:110,000,000 thus smaller values yield greater detail. Recognized
            values from most to least detailed include ``'10m'``, ``'50m'``,
            and ``'110m'``.
        cmap (matplotlib.colors.Colormap or str or None):
            The colormap for the plot.
        **kwargs:
            Forwarded to :meth:`xarray.DataArray.plot.contourf`.


    Returns:
        matplotlib.figure.Figure:
            The figure that was drawn.
    '''
    from apollo import nam

    # Select the feature
    data = xrds[feature]

    # Select along the reftime, forecast, and z dimensions.
    if 'forecast' in data.dims:
        data = data.isel(forecast=forecast)
    if 'reftime' in data.dims:
        if isinstance(reftime, int):
            data = data.isel(reftime=reftime)
        else:
            data = data.sel(reftime=reftime)
    if len(data.dims) == 3:
        z_dim = data.dims[0]
        data = data.isel({z_dim: level})

    # Get the axes.
    fig = plt.figure()
    ax = plt.axes(projection=nam.NAM218_PROJ)

    # Plot the data.
    contours = data.plot.contourf(ax=ax, transform=nam.NAM218_PROJ, cmap=cmap)

    # Draw the map.
    feature = MAP_FEATURES[detail].with_scale(scale)
    ax.add_feature(feature, edgecolor='black', facecolor='none')
    ax.set_global()
    ax.autoscale()

    # Set the title.
    if title is None:
        reftime_iso = apollo.Timestamp(data.reftime.data).isoformat()
        plt.title(f'{reftime_iso}Z + {forecast} hours')
    else:
        plt.title(title)

    return fig
