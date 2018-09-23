'''Additional vizualizations for matplotlib.
'''

import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1 import make_axes_locatable


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
            The axes on which the heatmap was drawn. This is set as the current
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
        index = pd.DatetimeIndex(start='2016-12-01', end='2019-05-28', freq='1D')
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
