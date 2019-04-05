from __future__ import print_function
from functools import partial
from ipywidgets import interact, Layout, Output
import ipywidgets as widgets
from IPython.display import display, FileLink, clear_output

import numpy as np

import osh5vis
import osh5io
import glob
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize, PowerNorm, SymLogNorm
from matplotlib import ticker
from matplotlib._pylab_helpers import Gcf as pylab_gcf
import subprocess
from datetime import datetime
import time
import re


print("Importing osh5visipy. Please use `%matplotlib notebook' in your jupyter/ipython notebook;")
print("use `%matplotlib widget' if you are using newer version of matplotlib (3.0) + jupyterlab (0.35)")


do_nothing = lambda x : x


def os2dplot_w(data, *args, pltfunc=osh5vis.osimshow, show=True, grid=None, **kwargs):
    """
    2D plot with widgets
    :param data: 2D H5Data
    :param args: arguments passed to 2d plotting widgets. reserved for future use
    :param show: whether to show the widgets
    :param kwargs: keyword arguments passed to 2d plotting widgets. reserved for future use
    :return: if show == True return None otherwise return a list of widgets
    """
    if isinstance(data, str):
        h5data = osh5io.read_h5(data)
        wl = Generic2DPlotCtrl(h5data, *args, pltfunc=pltfunc, **kwargs).widgets_list
    elif isinstance(data, (tuple, list)):
        if not grid:
            raise ValueError('Specify the grid layout when plotting more than one quantity!')
        if isinstance(data[0], str):
            data = [osh5io.read_h5(n) for n in data]
        wl = MultiPanelCtrl((Generic2DPlotCtrl,) * len(data), data, grid, **kwargs).widgets_list
    else:
        wl = Generic2DPlotCtrl(data, *args, pltfunc=pltfunc, **kwargs).widgets_list
    if show:
        display(*wl)
    else:
        return wl


osimshow_w = partial(os2dplot_w, pltfunc=osh5vis.osimshow)
oscontour_w = partial(os2dplot_w, pltfunc=osh5vis.oscontour)
oscontourf_w = partial(os2dplot_w, pltfunc=osh5vis.oscontourf)


def slicer_w(data, *args, show=True, slider_only=False, **kwargs):
    """
    A slider for 3D data
    :param data: 3D H5Data or directory name (a string)
    :param args: arguments passed to plotting widgets. reserved for future use
    :param show: whether to show the widgets
    :param slider_only: if True only show the slider otherwise show also other plot control (aka 'the tab')
    :param kwargs: keyword arguments passed to 2d plotting widgets. reserved for future use
    :return: whatever widgets that are not shown
    """
    if isinstance(data, str):
        wl = DirSlicer(data, *args, **kwargs).widgets_list
        tab, slider = wl[0], widgets.HBox(wl[1:-1])
    elif isinstance(data, (tuple, list)):
        if isinstance(data[0], (str, tuple, list)):
            wl = MPDirSlicer(data, *args, **kwargs).widgets_list
            tab, slider = wl[0], widgets.HBox(wl[1:-1])
    else:
        wl = Slicer(data, *args, **kwargs).widgets_list
        tab, slider = wl[0], widgets.HBox(wl[1:-1])
    if show:
        if slider_only:
            display(slider, wl[-1])
            return tab
        else:
            display(tab, slider, wl[-1])
    else:
        return wl


def animation_w(data, *args, **kwargs):
    wl = Animation(data, *args, **kwargs).widgets_list
    display(widgets.VBox([wl[0], widgets.HBox(wl[1:4]), widgets.HBox(wl[4:-2]), widgets.VBox(wl[-2:])]))


class FigureManager(object):
    def __init__(self):
        self.managers, self.figures = self.refresh()
        self.refreshbtn = widgets.Button(description='refresh', disabled=False, tooltip='get all currently opened figures', icon='refresh')
        self.deletebtn = widgets.Button(description='delete', tooltip='delete this figure', icon='trash', button_style='danger')
        self.selection = widgets.ToggleButtons(options=list(range(len(self.figures))), description='Figures:', value=None)
#                                                   layout=_items_layout, style={'description_width': 'initial', "button_width": 'initial'})
        self.display = Output()
        self.selection.observe(self.display_figure, 'value')
        self.refreshbtn.on_click(self.refresh_wgt)
        self.deletebtn.on_click(self.delete)
        self._widget = widgets.VBox([widgets.HBox([self.refreshbtn, self.deletebtn]), self.selection])
        with self.display:
            display(self._widget)

    def display_figure(self, change):
        self.display.clear_output(wait=True)
        with self.display:
            display(self._widget)
            if isinstance(change['new'], int):
                display(self.figures[change['new']])

    @property
    def widget(self):
        return self.display

#     @widget.setter
#     def widget(self, value):
#         self._widget = value

    def refresh(self):
        mngr = [manager for manager in pylab_gcf.get_all_fig_managers()]
        f = [m.canvas.figure for m in mngr]
        return mngr, f

    def refresh_wgt(self, *_):
        self.managers, self.figures = self.refresh()
        self.selection.options, self.selection.value = list(range(len(self.figures))), 0
#         self._widget = widgets.VBox([widgets.HBox([self.refreshbtn, self.deletebtn]), self.selection, self.display])

    def delete(self, *_):
        plt.close(self.figures[self.selection.value])
        self.refresh_wgt()


_items_layout = Layout(flex='1 1 auto', width='auto')
_get_delete_btn = lambda tp : widgets.Button(description='', tooltip='delete %s' % tp, icon='times', layout=Layout(width='32px'))

def _get_downloadable_url(filename):
    return '<a href="files%s" download="%s"> %s </a>' % (os.path.abspath(filename),
                                                         os.path.basename(filename), filename)


class Generic2DPlotCtrl(object):
    tab_contents = ['Data', 'Axes', 'Overlay', 'Colorbar', 'Save', 'Figure']
    eps = 1e-40
    colormaps_available = sorted(c for c in plt.colormaps() if not c.endswith("_r"))

    def __init__(self, data, pltfunc=osh5vis.osimshow, slcs=(slice(None, ), slice(None, )), title=None, norm='',
                 fig=None, figsize=None, time_in_title=True, cgs_time=False, ax=None, output_widget=None,
                 xlabel=None, ylabel=None, onDestruction=do_nothing,
                 convert_xaxis=False, convert_yaxis=False, **kwargs):
        self._data, self._slcs, self.im_xlt, self.time_in_title, self.pltfunc, self.onDestruction = \
        data, slcs, None, time_in_title, pltfunc, onDestruction
        user_cmap, show_colorbar = kwargs.pop('cmap', 'jet'), kwargs.pop('colorbar', True)
        tab = []
        # # # -------------------- Tab0 --------------------------
        # title
        if not title:
            title = osh5vis.default_title(data, show_time=False)
        t_in_axis = data.has_axis('t')
        self.if_reset_title = widgets.Checkbox(value=True, description='Auto', layout=_items_layout)
        self.datalabel = widgets.Text(value=title, placeholder='data', continuous_update=False,
                                     description='Data Name:', disabled=self.if_reset_title.value, layout=_items_layout)
        self.if_show_time = widgets.Checkbox(value=time_in_title and not t_in_axis, description='Time in title, ', layout=_items_layout)
        self.time_in_cgs = widgets.Checkbox(value=cgs_time, description='time in cgs unit',
                                            layout={'width': 'initial'}, style={'description_width': 'initial'})
        lognorm = norm == 'Log'
        self.__pp = np.abs if lognorm else do_nothing
        if 'clim' in kwargs:
            if kwargs['clim'][0] is not None:
                vmin, auto_vmin = kwargs['clim'][0], False
                logvmin = vmin if lognorm else self.eps
            else:
                vmin, logvmin, auto_vmin = np.min(data), self.eps, True
            if kwargs['clim'][1] is not None:
                vmax, auto_vmax = kwargs['clim'][1], False
            else:
                vmax, auto_vmax = np.max(data), True
        else:
            vmin, logvmin, vmax, auto_vmin, auto_vmax = np.min(data), self.eps, np.max(data), True, True
        # normalization
        # general parameters: vmin, vmax, clip
        self.if_vmin_auto = widgets.Checkbox(value=auto_vmin, description='Auto', layout=_items_layout, style={'description_width': 'initial'})
        self.if_vmax_auto = widgets.Checkbox(value=auto_vmax, description='Auto', layout=_items_layout, style={'description_width': 'initial'})
        self.vmin_wgt = widgets.FloatText(value=vmin, description='vmin:', continuous_update=False,
                                          disabled=self.if_vmin_auto.value, layout=_items_layout, style={'description_width': 'initial'})
        self.vlogmin_wgt = widgets.FloatText(value=logvmin, description='vmin:', continuous_update=False,
                                             disabled=self.if_vmin_auto.value, layout=_items_layout, style={'description_width': 'initial'})
        self.vmax_wgt = widgets.FloatText(value=vmax, description='vmax:', continuous_update=False,
                                          disabled=self.if_vmax_auto.value, layout=_items_layout, style={'description_width': 'initial'})
        self.if_clip_cm = widgets.Checkbox(value=True, description='Clip', layout=_items_layout, style={'description_width': 'initial'})
        # PowerNorm specific
        self.gamma = widgets.FloatText(value=1, description='gamma:', continuous_update=False,
                                       layout=_items_layout, style={'description_width': 'initial'})
        # SymLogNorm specific
        self.linthresh = widgets.FloatText(value=self.eps, description='linthresh:', continuous_update=False,
                                           layout=_items_layout, style={'description_width': 'initial'})
        self.linscale = widgets.FloatText(value=1.0, description='linscale:', continuous_update=False,
                                          layout=_items_layout, style={'description_width': 'initial'})

        # build the widgets tuple
        ln_wgt = (LogNorm, widgets.VBox([widgets.HBox([self.vmax_wgt, self.if_vmax_auto]),
                                         widgets.HBox([self.vlogmin_wgt, self.if_vmin_auto]), self.if_clip_cm]))
        n_wgt = (Normalize, widgets.VBox([widgets.HBox([self.vmax_wgt, self.if_vmax_auto]),
                                          widgets.HBox([self.vmin_wgt, self.if_vmin_auto]), self.if_clip_cm]))
        pn_wgt = (PowerNorm, widgets.VBox([widgets.HBox([self.vmax_wgt, self.if_vmax_auto]),
                                           widgets.HBox([self.vmin_wgt, self.if_vmin_auto]), self.if_clip_cm,
                                           self.gamma]))
        sln_wgt = (SymLogNorm, widgets.VBox(
            [widgets.HBox([self.vmax_wgt, self.if_vmax_auto]),
             widgets.HBox([self.vmin_wgt, self.if_vmin_auto]), self.if_clip_cm, self.linthresh, self.linscale]))

        # find out default value for norm_selector
        norm_avail = {'Log': ln_wgt, 'Normalize': n_wgt, 'Power': pn_wgt, 'SymLog': sln_wgt}
        self.norm_selector = widgets.Dropdown(options=norm_avail, style={'description_width': 'initial'},
                                              value=norm_avail.get(norm, n_wgt), description='Normalization:')
        self.__old_norm = self.norm_selector.value
        # additional care for LorNorm()
        self.__handle_lognorm()
        # re-plot button
        self.norm_btn_wgt = widgets.Button(description='Apply', disabled=False, tooltip='Update normalization', icon='refresh')
        tab.append(self.get_tab_data())

        # # # -------------------- Tab1 --------------------------
        self.setting_instructions = widgets.Label(value="Enter invalid value to reset", layout=_items_layout)
        self.apply_range_btn = widgets.Button(description='Apply', disabled=False, \
                                              tooltip='set range. (* this will delete all overlaid plots *)', icon='refresh')
        self.axis_lim_wgt = widgets.HBox([self.setting_instructions, self.apply_range_btn])
        # x axis
        xmin, xmax, xinc, ymin, ymax, yinc = self.__get_xy_minmax_delta()
        if convert_xaxis:
            self.xconv, xunit = data.axes[1].punit_convert_factor()
        else:
            self.xconv, xunit = 1.0, data.axes[1].units
        self.x_min_wgt = widgets.BoundedFloatText(value=xmin * self.xconv, min=xmin * self.xconv, max=xmax * self.xconv, step=xinc * self.xconv/2, description='xmin:',
                                                  continuous_update=False, layout=_items_layout, style={'description_width': 'initial'})
        self.x_max_wgt = widgets.BoundedFloatText(value=xmax * self.xconv, min=xmin * self.xconv, max=xmax * self.xconv, step=xinc * self.xconv/2, description='xmax:',
                                                  continuous_update=False, layout=_items_layout, style={'description_width': 'initial'})
        self.x_step_wgt = widgets.BoundedFloatText(value=xinc * self.xconv, step=xinc * self.xconv, continuous_update=False,
                                            description='$\Delta x$:', layout=_items_layout, style={'description_width': 'initial'})
        widgets.jslink((self.x_min_wgt, 'max'), (self.x_max_wgt, 'value'))
        widgets.jslink((self.x_max_wgt, 'min'), (self.x_min_wgt, 'value'))
        # x label
        self.if_reset_xlabel = widgets.Checkbox(value=True, description='Auto', layout=_items_layout, style={'description_width': 'initial'})
        self.if_x_cgs_unit = widgets.Checkbox(value=False, description='cgs unit;  ', layout={'width': 'initial'}, style={'description_width': 'initial'})
        if xlabel is False:
            self._xlabel = None
        elif isinstance(xlabel, str):
            self._xlabel = xlabel
        else:
            self._xlabel = osh5vis.axis_format(data.axes[1].long_name, xunit)
        self.xlabel = widgets.Text(value=self._xlabel,
                                   placeholder='x', continuous_update=False,
                                   description='X label:', disabled=self.if_reset_xlabel.value)
        widgets.jslink((self.xlabel, 'disabled'), (self.if_reset_xlabel, 'value'))
        self.xaxis_lim_wgt = widgets.HBox([self.if_x_cgs_unit, self.x_min_wgt, self.x_max_wgt, self.x_step_wgt,
                                           widgets.HBox([self.xlabel, self.if_reset_xlabel], layout=Layout(border='solid 1px'))])
        # y axis
        if convert_yaxis:
            self.yconv, yunit = data.axes[0].punit_convert_factor()
        else:
            self.yconv, yunit = 1.0, data.axes[0].units
        self.y_min_wgt = widgets.BoundedFloatText(value=ymin * self.yconv, min=ymin * self.yconv, max=ymax * self.yconv, step=yinc * self.yconv/2, description='ymin:',
                                                  continuous_update=False, layout=_items_layout, style={'description_width': 'initial'})
        self.y_max_wgt = widgets.BoundedFloatText(value=ymax * self.yconv, min=ymin * self.yconv, max=ymax * self.yconv, step=yinc * self.yconv/2, description='ymax:',
                                                  continuous_update=False, layout=_items_layout, style={'description_width': 'initial'})
        self.y_step_wgt = widgets.BoundedFloatText(value=yinc * self.yconv, step=yinc * self.yconv, continuous_update=False, description='$\Delta y$:',
                                                   layout=_items_layout, style={'description_width': 'initial'})
        widgets.jslink((self.y_min_wgt, 'max'), (self.y_max_wgt, 'value'))
        widgets.jslink((self.y_max_wgt, 'min'), (self.y_min_wgt, 'value'))
        # y label
        self.if_reset_ylabel = widgets.Checkbox(value=True, description='Auto', layout=_items_layout, style={'description_width': 'initial'})
        self.if_y_cgs_unit = widgets.Checkbox(value=False, description='cgs unit;  ', layout={'width': 'initial'}, style={'description_width': 'initial'})
        if ylabel is False:
            self._ylabel = None
        elif isinstance(ylabel, str):
            self._ylabel = ylabel
        else:
            self._ylabel = osh5vis.axis_format(data.axes[0].long_name, yunit)
        self.ylabel = widgets.Text(value=self._ylabel,
                                   placeholder='y', continuous_update=False,
                                   description='Y label:', disabled=self.if_reset_ylabel.value)
        widgets.jslink((self.ylabel, 'disabled'), (self.if_reset_ylabel, 'value'))
        self.yaxis_lim_wgt = widgets.HBox([self.if_y_cgs_unit, self.y_min_wgt, self.y_max_wgt, self.y_step_wgt,
                                           widgets.HBox([self.ylabel, self.if_reset_ylabel], layout=Layout(border='solid 1px'))])
        tab.append(widgets.VBox([self.axis_lim_wgt, self.xaxis_lim_wgt, self.yaxis_lim_wgt]))

        # # # -------------------- Tab2 --------------------------
        overlay_item_layout = Layout(display='flex', flex_flow='row wrap', width='auto')
        self.__analysis_def = {'Average': {'Simple': lambda x, a : np.mean(x, axis=a), 'RMS': lambda x, a : np.sqrt(np.mean(x*x, axis=a))},
                               'Sum': {'Simple': lambda x, a : np.sum(x, axis=a), 'Square': lambda x, a : np.sum(x*x, axis=a),
                                       'ABS': lambda x, a : np.sum(np.abs(x, axis=a))},
                               'Min': {'Simple': lambda x, a : np.min(x, axis=a), 'ABS': lambda x, a : np.min(np.abs(x), axis=a)},
                               'Max': {'Simple': lambda x, a : np.max(x, axis=a), 'ABS': lambda x, a : np.max(np.abs(x), axis=a)}} #TODO: envelope
        analist = [k for k in self.__analysis_def.keys()]
        # x lineout
        self.xlineout_wgt = widgets.BoundedFloatText(value=ymin, min=ymin, max=ymax, style={'description_width': 'initial'},
                                                     step=yinc, description=self.ylabel.value, layout={'width': 'initial'})
        widgets.jslink((self.xlineout_wgt, 'description'), (self.ylabel, 'value'))
        widgets.jslink((self.xlineout_wgt, 'min'), (self.y_min_wgt, 'value'))
        widgets.jslink((self.xlineout_wgt, 'max'), (self.y_max_wgt, 'value'))
        widgets.jslink((self.xlineout_wgt, 'step'), (self.y_step_wgt, 'value'))
        self.add_xlineout_btn = widgets.Button(description='Lineout', tooltip='Add x-lines', layout={'width': 'initial'})
        # simple analysis in x direction
        self.xananame = widgets.Dropdown(options=analist, value=analist[0], description='Analysis:',
                                         layout={'width': 'initial'}, style={'description_width': 'initial'})
        xanaoptlist = [k for k in self.__analysis_def[analist[0]].keys()]
        self.xanaopts = widgets.Dropdown(options=xanaoptlist, value=xanaoptlist[0], description='',
                                         layout={'width': 'initial'}, style={'description_width': 'initial'})
        self.anaxmin = widgets.BoundedFloatText(value=ymin, min=ymin, max=ymax, step=yinc, description='from',
                                                layout={'width': 'initial'}, style={'description_width': 'initial'})
        self.anaxmax = widgets.BoundedFloatText(value=ymax, min=ymin, max=ymax, step=yinc, description='to',
                                                layout={'width': 'initial'}, style={'description_width': 'initial'})
        widgets.jslink((self.anaxmin, 'min'), (self.y_min_wgt, 'value'))
        widgets.jslink((self.anaxmin, 'max'), (self.anaxmax, 'value'))
        widgets.jslink((self.anaxmin, 'step'), (self.y_step_wgt, 'value'))
        widgets.jslink((self.anaxmax, 'min'), (self.anaxmin, 'value'))
        widgets.jslink((self.anaxmax, 'max'), (self.y_max_wgt, 'value'))
        widgets.jslink((self.anaxmax, 'step'), (self.y_step_wgt, 'value'))
        self.xana_add = widgets.Button(description='Add', tooltip='Add analysis as x line plot', layout={'width': 'initial'})
        xlinegroup = widgets.HBox([self.xananame, self.xanaopts, self.anaxmin, self.anaxmax, self.xana_add], layout=Layout(border='solid 1px'))
        # list of x lines plotted
        self.xlineout_list_wgt = widgets.Box(children=[], layout=overlay_item_layout, style={'description_width': 'initial'})
        self.xlineout_tab = widgets.VBox([widgets.HBox([widgets.HBox([self.xlineout_wgt, self.add_xlineout_btn],
                                                                     layout=Layout(border='solid 1px', flex='1 1 auto', width='auto')),
                                                        xlinegroup]), self.xlineout_list_wgt])
        # y lineout
        self.ylineout_wgt = widgets.BoundedFloatText(value=xmin, min=xmin, max=xmax, style={'description_width': 'initial'},
                                                     step=xinc, description=self.xlabel.value, layout={'width': 'initial'})
        widgets.jslink((self.ylineout_wgt, 'description'), (self.xlabel, 'value'))
        widgets.jslink((self.ylineout_wgt, 'min'), (self.x_min_wgt, 'value'))
        widgets.jslink((self.ylineout_wgt, 'max'), (self.x_max_wgt, 'value'))
        widgets.jslink((self.ylineout_wgt, 'step'), (self.x_step_wgt, 'value'))
        self.add_ylineout_btn = widgets.Button(description='Lineout', tooltip='Add y-lines', layout={'width': 'initial'})
        # simple analysis in x direction
        self.yananame = widgets.Dropdown(options=analist, value=analist[0], description='Analysis:',
                                         layout={'width': 'initial'}, style={'description_width': 'initial'})
        yanaoptlist = [k for k in self.__analysis_def[analist[0]].keys()]
        self.yanaopts = widgets.Dropdown(options=yanaoptlist, value=yanaoptlist[0], description='',
                                         layout={'width': 'initial'}, style={'description_width': 'initial'})
        self.anaymin = widgets.BoundedFloatText(value=xmin, min=xmin, max=xmax, step=xinc, description='from',
                                                layout={'width': 'initial'}, style={'description_width': 'initial'})
        self.anaymax = widgets.BoundedFloatText(value=xmax, min=xmin, max=xmax, step=xinc, description='to',
                                                layout={'width': 'initial'}, style={'description_width': 'initial'})
        widgets.jslink((self.anaymin, 'min'), (self.x_min_wgt, 'value'))
        widgets.jslink((self.anaymin, 'max'), (self.anaymax, 'value'))
        widgets.jslink((self.anaymin, 'step'), (self.x_step_wgt, 'value'))
        widgets.jslink((self.anaymax, 'min'), (self.anaymin, 'value'))
        widgets.jslink((self.anaymax, 'max'), (self.x_max_wgt, 'value'))
        widgets.jslink((self.anaymax, 'step'), (self.x_step_wgt, 'value'))
        self.yana_add = widgets.Button(description='Add', tooltip='Add analysis as y line plot', layout={'width': 'initial'})
        ylinegroup = widgets.HBox([self.yananame, self.yanaopts, self.anaymin, self.anaymax, self.yana_add],
                                  layout=Layout(width='initial', border='solid 1px'))
        # list of x lines plotted
        self.ylineout_list_wgt = widgets.Box(children=[], layout=overlay_item_layout)
        self.ylineout_tab = widgets.VBox([widgets.HBox([widgets.HBox([self.ylineout_wgt, self.add_ylineout_btn],
                                                                     layout=Layout(border='solid 1px', flex='1 1 auto', width='auto')),
                                                        ylinegroup]), self.ylineout_list_wgt])
#         self.ct_alpha = widgets.BoundedFloatText(value=1.0, min=0., max=1., step=0.01, layout={'width': 'initial'}, style={'description_width': 'initial'})
        self.ct_auto_color = widgets.ToggleButtons(options=['colormap', 'manual', 'same'], description='Color:', value='colormap',
                                                   tooltips=['use selected colormap', 'set each level indevidually', 'monochromatic, same as the last level'],
                                                   layout=_items_layout, style={'description_width': 'initial', "button_width": 'initial'})
        self.ct_antialiased = widgets.Checkbox(value=False, description='antialiased; ', layout=_items_layout, style={'description_width': 'initial'})
        self.ct_if_clabel = widgets.Checkbox(value=False, description='clabels ', layout=_items_layout, style={'description_width': 'initial'})
        self.ct_if_inline_clabel = widgets.Checkbox(value=False, description='inline; ', disabled=True,
                                                    layout=_items_layout, style={'description_width': 'initial'})
        self.ct_cmap_selector = widgets.Dropdown(options=self.colormaps_available, value=user_cmap, description='  colormap:',
                                                 disabled=(self.ct_auto_color.value != 'colormap'), layout={'width': 'initial'}, style={'description_width': 'initial'})
        self.ct_cmap_reverse = widgets.Checkbox(value=False, description='Reverse; ', disabled=self.ct_cmap_selector.disabled,
                                                layout=_items_layout, style={'description_width': 'initial'})
        widgets.jslink((self.ct_cmap_reverse, 'disabled'), (self.ct_cmap_selector, 'disabled'))
        self.ct_method = widgets.Dropdown(options=['contour', 'contourf'], value='contour', description='', layout=_items_layout, style={'description_width': 'initial'})
        self.ct_plot_btn = widgets.Button(description='Plot', tooltip='plot overlay', style={'description_width': 'initial'}, layout={'width': 'initial'})
        self.ct_opts_list, self.ct_wgt_list = widgets.Box(children=[], layout=overlay_item_layout), widgets.Box(children=[], layout=overlay_item_layout)
        self.ct_opts_dict, self.ct_plot_dict = {}, {}  # dict to keep track of the overlaid plots
        self.ct_num_levels_opts = widgets.ToggleButtons(options=['auto', 'option', 'fixed:'], description=' number of levels:', value='auto',
                                                        tooltips=['let the plotter decide how many levels to plot', 'same number of levels as the level options added below',
                                                                  'excact number of levels (some of the level options added below may not be used)'],
                                                        layout=_items_layout, style={'description_width': 'initial', "button_width": 'initial'})
        self.ct_num_levels = widgets.BoundedIntText(value=8, min=1, max=plt.MaxNLocator.MAXTICKS, description='', disabled=(self.ct_num_levels_opts.value != 'fixed:'),
                                                    layout={'width': 'initial'}, style={'description_width': 'initial'})
        self.ct_level = widgets.Text(value='0.0', placeholder='0.0, 0.01, 10', description='Levels=', disabled=(self.ct_num_levels_opts.value != 'option'),
                                     layout=_items_layout, style={'description_width': 'initial'})
        self.ct_colorpicker = widgets.ColorPicker(concise=False, description='; color:', value='black', disabled=(self.ct_auto_color.value == 'colormap'),
                                                  style={'description_width': 'initial'}, layout={'width': '200px'})
        self.ct_linestyle = widgets.Dropdown(options=[None, 'solid', 'dashed', 'dashdot', 'dotted'], value=None, description='; linestyle:',
                                             style={'description_width': 'initial'}, layout={'width': 'initial'})
        self.ct_add_lvl_btn = widgets.Button(description='Add', tooltip='Add level options. Will use default settings if no option is added',
                                             style={'description_width': 'initial'}, layout={'width': 'initial'})
        self.ct_info_output = Output()
        self.ct_tab = widgets.VBox([widgets.HBox([self.ct_antialiased, self.ct_if_clabel, self.ct_if_inline_clabel, self.ct_auto_color,
                                                  self.ct_cmap_selector, self.ct_cmap_reverse, self.ct_num_levels_opts, self.ct_num_levels]),
                                    widgets.HBox([self.ct_level, self.ct_colorpicker, self.ct_add_lvl_btn, self.ct_method, self.ct_linestyle, self.ct_plot_btn]),
                                    self.ct_opts_list, self.ct_info_output, self.ct_wgt_list])

        self.overlay = widgets.Tab(children=[self.xlineout_tab, self.ylineout_tab, self.ct_tab])
        [self.overlay.set_title(i, tt) for i, tt in enumerate(['x-lines', 'y-lines', 'contours'])]
        tab.append(self.overlay)

        # # # -------------------- Tab3 --------------------------
        self.colorbar = widgets.Checkbox(value=show_colorbar, description='Show colorbar')
        self.cmap_selector = widgets.Dropdown(options=self.colormaps_available, value=user_cmap,
                                              description='Colormap:', disabled=not show_colorbar)
        self.cmap_reverse = widgets.Checkbox(value=False, description='Reverse', disabled=not show_colorbar)
        # colorbar
        self.if_reset_cbar = widgets.Checkbox(value=True, description='Auto', disabled=not show_colorbar)
        self.cbar = widgets.Text(value=data.units.tex(), placeholder='a.u.', continuous_update=False,
                                 description='Colorbar:', disabled=self.if_reset_cbar.value or not show_colorbar)
        tab.append(widgets.VBox([self.colorbar,
                                 widgets.HBox([self.cmap_selector, self.cmap_reverse], layout=_items_layout),
                                 widgets.HBox([self.cbar, self.if_reset_cbar])], layout=_items_layout))

        # # # -------------------- Tab4 --------------------------
        self.saveas = widgets.Button(description='Save current plot', tooltip='save current plot', button_style='')
        self.dlink = widgets.HTML(value='', description='')
        self.figname = widgets.Text(value='figure.eps', description='Figure name:', placeholder='figure name')
        self.dpi = widgets.BoundedIntText(value=300, min=4, max=3000, description='DPI:')
        tab.append(self.get_tab_save())

        # # # -------------------- Tab5 --------------------------
        width, height = figsize or plt.rcParams.get('figure.figsize')
        self.figwidth = widgets.BoundedFloatText(value=width, min=0.1, step=0.01, description='Width:')
        self.figheight = widgets.BoundedFloatText(value=height, min=0.1, step=0.01, description='Height:')
        self.resize_btn = widgets.Button(description='Adjust figure', tooltip='Update figure', icon='refresh')
        self.destroy_fig_btn = widgets.Button(description='Close figure', tooltip='close figure and destroy this widget', icon='trash')
        tab.append(widgets.VBox([widgets.HBox([self.figwidth, self.figheight, self.resize_btn], layout=_items_layout),
                                 self.destroy_fig_btn]))

        # construct the tab
        self.tab = widgets.Tab(layout=_items_layout)
        self.tab.children = tab
        [self.tab.set_title(i, tt) for i, tt in enumerate(self.tab_contents)]

        # plotting and then setting normalization colors
        self.out_main = output_widget or Output()
        self.observer_thrd, self.cb = None, None
#         if not fig:
#             ax = None
        with self.out_main:
            self.fig = fig or plt.figure(figsize=[width, height], constrained_layout=True)
            self.ax = ax or self.fig.add_subplot(111)
            self.im, self.cb = self.plot_data()
#             vmin, vmax = self.__get_vminmax(from_widgets=True)
            self.im.set_clim(vmin, vmax)
        # cannot get the axes ticks formatter here because the figure may not have been shown
        self.xfmttr, self.yfmttr = None, None
#             plt.show()
        self.axx, self.axy, self._xlineouts, self._ylineouts, self.im2 = None, None, {}, {}, []

        # link and activate the widgets
        self.if_reset_title.observe(self.__update_title, 'value')
        self.if_reset_xlabel.observe(self.__update_xlabel, 'value')
        self.if_reset_ylabel.observe(self.__update_ylabel, 'value')
        self.if_x_cgs_unit.observe(self._update_xconverter, 'value')
        self.if_y_cgs_unit.observe(self._update_yconverter, 'value')
        self.if_reset_cbar.observe(self.__update_cbar, 'value')
        self.norm_btn_wgt.on_click(self.update_norm)
        self.if_vmin_auto.observe(self.__update_vmin, 'value')
        self.if_vmax_auto.observe(self.__update_vmax, 'value')
        self.norm_selector.observe(self.__update_norm_wgt, 'value')
        self.cmap_selector.observe(self.update_cmap, 'value')
        self.cmap_reverse.observe(self.update_cmap, 'value')
        self.datalabel.observe(self.update_title, 'value')
        self.if_show_time.observe(self.update_title, 'value')
        self.time_in_cgs.observe(self.update_title, 'value')
        self.xlabel.observe(self.update_xlabel, 'value')
        self.ylabel.observe(self.update_ylabel, 'value')
        self.cbar.observe(self.update_cbar, 'value')
        self.apply_range_btn.on_click(self.update_plot_area)
        self.figname.observe(self.__reset_save_button, 'value')
        self.saveas.on_click(self.__try_savefig)
        self.colorbar.observe(self.__toggle_colorbar, 'value')
        self.resize_btn.on_click(self.adjust_figure)
        self.add_xlineout_btn.on_click(self.__add_xlineout)
        self.add_ylineout_btn.on_click(self.__add_ylineout)
        self.xananame.observe(self.__update_xanaopts, 'value')
        self.xana_add.on_click(self.__add_xana)
        self.yananame.observe(self.__update_yanaopts, 'value')
        self.yana_add.on_click(self.__add_yana)
        self.destroy_fig_btn.on_click(self.self_destruct)
        self.ct_auto_color.observe(self._on_ct_auto_color_wgt_change, 'value')
        self.ct_num_levels_opts.observe(self._on_ct_num_lvl_opts_wgt_change, 'value')
        self.ct_add_lvl_btn.on_click(self._add_contour_lvl_opts)
        self.ct_plot_btn.on_click(self._add_contour_plot)
        self.ct_if_clabel.observe(self._on_clabel_toggle, 'value')
        self.ct_method.observe(self._on_ct_method_change, 'value')

        self.vmin_wgt.value, self.vlogmin_wgt.value, self.vmax_wgt.value = vmin, logvmin, vmax

    @property
    def widgets_list(self):
        return self.tab, self.out_main

    @property
    def widget(self):
        return widgets.VBox([self.tab, self.out_main])

    def get_dataname(self):
        return self._data.name

    def get_time_label(self, convert_tunit=False):
        return osh5vis.time_format(self._data.run_attrs['TIME'][0], self._data.run_attrs['TIME UNITS'], convert_tunit=convert_tunit)

    def update_data(self, data, slcs):
        self._data, self._slcs = data, slcs
        self._xlabel, self._ylabel = osh5vis.axis_format(data.axes[1].long_name, data.axes[1].units), \
                                     osh5vis.axis_format(data.axes[0].long_name, data.axes[0].units)
        self.__update_title()
        self.__update_xlabel({'new': True})
        self.__update_ylabel({'new': True})

    def reset_plot_area(self):
        self.x_min_wgt.value, self.x_max_wgt.value, xstep, \
        self.y_min_wgt.value, self.y_max_wgt.value, ystep = self.__get_xy_minmax_delta()
        self.x_step_wgt.value, self.y_step_wgt.value = xstep / 2, ystep / 2
        self.__destroy_all_xlineout()
        self.__destroy_all_ylineout()
        self._ct_destroy_all()

    def redraw(self, data, update_vminmax=False):
        if self.pltfunc is osh5vis.osimshow:
            "if the size of the data is the same we can just redraw part of figure"
            self._data = data
            processed_data = self.__pp(data[self._slcs])
            self.im.set_data(processed_data)
            if update_vminmax and (self.if_vmax_auto.value or self.if_vmin_auto.value):
                vmin = np.min(processed_data) if self.if_vmin_auto.value else None
                vmax = np.max(processed_data) if self.if_vmax_auto.value else None
                self.im.set_clim(vmin, vmax)
            self.fig.canvas.draw_idle()
        else:
            "for contour/contourf we have to do a full replot"
            self._data = data
            for col in self.im.collections:
                col.remove()
            self.replot_axes()

    def update_title(self, *_):
        self.ax.axes.set_title(self.get_plot_title())

    def update_xlabel(self, change):
        self.ax.axes.xaxis.set_label_text(change['new'])

    def update_ylabel(self, change):
        self.ax.axes.yaxis.set_label_text(change['new'])

    def update_cbar(self, change):
        self.im.colorbar.set_label(change['new'])

    def update_cmap(self, _change):
        cmap = self.cmap_selector.value if not self.cmap_reverse.value else self.cmap_selector.value + '_r'
        self.im.set_cmap(cmap)
        self.cb.set_cmap(cmap)

#     def update_time_label(self):
#         self._time = osh5vis.time_format(self._data.run_attrs['TIME'][0], self._data.run_attrs['TIME UNITS'])

    def adjust_figure(self, *_):
        with self.out_main:
            self.out_main.clear_output(wait=True)
            # this dosen't work in all scenarios. it could be a bug in matplotlib/jupyterlab
            self.fig.set_size_inches(self.figwidth.value, self.figheight.value)

    def replot_axes(self):
#         self.fig.delaxes(self.ax)
# #         self.fig.clear()
#         self.ax = self.fig.add_subplot(111)
        self.ax.cla()
#         self.im.remove()
        self.im, cb = self.plot_data(colorbar=self.colorbar.value)
        if self.colorbar.value:
            self.cb.remove()
            self.cb = cb
#         self.fig.subplots_adjust()  # does not compatible with constrained_layout in Matplotlib 3.0

    def __get_xy_minmax_delta(self):
        return (float('%.2g' % self._data.axes[1].min), float('%.2g' % self._data.axes[1].ax.max()), float('%.2g' % self._data.axes[1].increment),
                float('%.2g' % self._data.axes[0].min), float('%.2g' % self._data.axes[0].ax.max()), float('%.2g' % self._data.axes[0].increment))

    def _update_xy_minmaxstep_wgt(self, d):
        xmin, xmax, xstep, ymin, ymax, ystep = self.__get_xy_minmax_delta()
        if d == 'x':
            self.x_min_wgt.value, self.x_max_wgt.value, self.x_step_wgt.value = xmin * self.xconv, xmax * self.xconv, xstep * self.xconv * 0.5
        else:
            self.y_min_wgt.value, self.y_max_wgt.value, self.y_step_wgt.value = ymin * self.yconv, ymax * self.yconv, ystep * self.yconv * 0.5

    def update_plot_area(self, *_):
        bnd = [(self.y_min_wgt.value / self.yconv, self.y_max_wgt.value / self.yconv, self.y_step_wgt.value / self.yconv),
               (self.x_min_wgt.value / self.xconv, self.x_max_wgt.value / self.xconv, self.x_step_wgt.value / self.xconv)]
        self._slcs = tuple(slice(*self._data.get_index_slice(self._data.axes[i], bd)) for i, bd in enumerate(bnd))
        #TODO: maybe we can keep some of the overlaid plots but replot_axes will generate new axes.
        # for now delete everything for simplicity
        self.__destroy_all_xlineout()
        self.__destroy_all_ylineout()
        self._ct_destroy_all()
        self.replot_axes()

    def refresh_tab_wgt(self, update_list):
        """
        the tab.children is a tuple so we have to reconstruct the whole tab widget when
        addition/deletion of children widgets happens
        """
        tmp = self.tab.children
        newtab = [tmp[i] if not t else t for i, t in enumerate(update_list)]
        self.tab.children = tuple(newtab)

    def plot_data(self, **passthrough):
        ifcolorbar = passthrough.pop('colorbar', self.colorbar.value)
        return self.pltfunc(self.__pp(self._data[self._slcs]), cmap=self.cmap_selector.value,
                            norm=self.norm_selector.value[0](**self.__get_norm()), title=self.get_plot_title(),
                            xlabel=self.xlabel.value, ylabel=self.ylabel.value, cblabel=self.cbar.value,
                            ax=self.ax, fig=self.fig, colorbar=ifcolorbar,
                            convert_xaxis=self.if_x_cgs_unit.value, convert_yaxis=self.if_y_cgs_unit.value, **passthrough)

    def self_destruct(self, *_):
        plt.close(self.fig)
        for w in self.widgets_list:
            w.close()
        self.onDestruction()

    def get_plot_title(self):
        if self.datalabel.value:
            return self.datalabel.value + ((', ' + self.get_time_label(self.time_in_cgs.value)) if self.if_show_time.value else '')
        else:
            return self.get_time_label(self.time_in_cgs.value) if self.if_show_time.value else ''

    def get_tab_data(self):
        return widgets.HBox([widgets.VBox([self.norm_selector, self.norm_selector.value[1]]), self.norm_btn_wgt,
                             widgets.VBox([widgets.HBox([self.datalabel, self.if_reset_title]),
                                           widgets.HBox([self.if_show_time, self.time_in_cgs])])])

    def get_tab_save(self):
        return widgets.VBox([widgets.HBox([self.figname, self.dpi, self.saveas], layout=_items_layout),
                             self.dlink], layout=_items_layout)

    def extract_lineout_params(self, s):
        l = s.split()
        # we put all params in the tooltip of the delete button
        if "~" in l:  # this is analysis
            return int(l[2]), int(l[4]), l[5], l[6]
        else:
            return (float(l[1]), )

    def update_lineouts(self, dim='xy', description_only=False):
        # update x lineouts/analysis
        if 'x' in dim:
            for wgt in self.xlineout_list_wgt.children:
                cpk, nw = wgt.children
                params = self.extract_lineout_params(nw.tooltip)
                if len(params) > 1:
                    data, posstr, _ = self._get_xana_data_descr(*params, description_only=description_only)
                else:
                    data, posstr, _ = self.get_xlineout_data_and_index(params[0])
                if description_only:
                    cpk.description = posstr
                else:
                    self._xlineouts[cpk].set_ydata(data)
        # update y lineouts/analysis
        if 'y' in dim:
            for wgt in self.ylineout_list_wgt.children:
                cpk, nw = wgt.children
                params = self.extract_lineout_params(nw.tooltip)
                if len(params) > 1:
                    data, posstr, _ = self._get_yana_data_descr(*params, description_only=description_only)
                else:
                    data, posstr, _ = self.get_ylineout_data_and_index(params[0])
                if description_only:
                    cpk.description = posstr
                else:
                    self._ylineouts[cpk].set_xdata(data)

    def _on_clabel_toggle(self, change):
        self.ct_if_inline_clabel.disabled = not change['new']

    def _update_xconverter(self, change):
        if self.xfmttr is None:
            self.xfmttr = self.ax.xaxis.get_major_formatter()
        if change['new']:
            self.xconv, xunit = self._data.axes[1].punit_convert_factor()
        else:
            self.xconv, xunit = 1.0, self._data.axes[1].units
        self.ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: self.xfmttr(x*self.xconv, pos)))
        self._update_xy_minmaxstep_wgt(d='x')
        self.update_lineouts(dim='y', description_only=True)
        if self.if_reset_xlabel.value:
            self._xlabel = osh5vis.axis_format(self._data.axes[1].long_name, xunit)
            # toggle the reset checkbox
            self.if_reset_xlabel.value = False
            self.if_reset_xlabel.value = True

    def _update_yconverter(self, change):
        if self.yfmttr is None:
            self.yfmttr = self.ax.yaxis.get_major_formatter()
        if change['new']:
            self.yconv, yunit = self._data.axes[0].punit_convert_factor()
        else:
            self.yconv, yunit = 1.0, self._data.axes[0].units
        self.ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, pos: self.yfmttr(y*self.yconv, pos)))
        self._update_xy_minmaxstep_wgt(d='y')
        self.update_lineouts(dim='x', description_only=True)
        if self.if_reset_ylabel.value:
            self._ylabel = osh5vis.axis_format(self._data.axes[0].long_name, yunit)
            # toggle the reset checkbox
            self.if_reset_ylabel.value = False
            self.if_reset_ylabel.value = True

    def _on_ct_auto_color_wgt_change(self, change):
        if change['new'] == 'colormap':
            self.ct_cmap_selector.disabled, self.ct_colorpicker.disabled = False, True
        else:
            self.ct_cmap_selector.disabled, self.ct_colorpicker.disabled = True, False

    def _on_ct_method_change(self, change):
        # linestyle keyword only applies to contour
        self.ct_linestyle.disabled = change['new'] != 'contour'

    def _on_ct_num_lvl_opts_wgt_change(self, change):
        self.ct_num_levels.disabled = False if change['new'] == 'fixed:' else True
        self.ct_level.disabled = True if change['new'] != 'option' else False

    def _get_contour_opt_wgt(self, lvl, color, txt_wgt_width='initial', cp_wgt_width='initial'):
        return (widgets.FloatText(value=lvl, description='level=', disabled=(self.ct_num_levels_opts.value != 'option'),
                                  style={'description_width': 'initial'}, layout={'width': txt_wgt_width}),
                widgets.ColorPicker(concise=False, description='; color:', value=color, disabled=(self.ct_auto_color.value == 'colormap'),
                                    style={'description_width': 'initial'}, layout={'width': cp_wgt_width}), _get_delete_btn('level %f' % lvl))

    def _remove_ct_lvl_opt(self, btn):
        lvl_wgt = self.ct_opts_dict.pop(btn)
        # remove level widgets from ct_opts_list
        tmp = list(self.ct_opts_list.children)
        tmp.remove(lvl_wgt)
        self.ct_opts_list.children = tuple(tmp)
        lvl_wgt.close()

    def _print_ct_info(self, msg, timeout=8):
        with self.ct_info_output:
            print(msg)
            time.sleep(timeout)
            self.ct_info_output.clear_output()

    def _extract_ct_kwargs_from_wgt(self):
        levels, colors = [], []
        for w in self.ct_opts_list.children:
            lvl, cp, db = w.children
            levels.append(lvl.value)
            colors.append(cp.value)
            lvl.close()
            cp.close()
            db.close()
            w.close()
        kwargs = {'antialiased': self.ct_antialiased.value}
        if self.ct_auto_color.value == 'colormap':
            kwargs['colors'] = None
            kwargs['cmap'] = self.ct_cmap_selector.value + ('_r' if self.ct_cmap_reverse.value else '')
            kwargs['norm'] = self.norm_selector.value[0](**self.__get_norm())
        elif self.ct_auto_color.value == 'manual':
            kwargs['colors'] = colors or None
        else:
            kwargs['colors'] = None if not colors else (colors[-1], ) * len(colors)

        if not self.ct_linestyle.disabled:
            kwargs['linestyles'] = self.ct_linestyle.value

        if self.ct_num_levels_opts.value == 'fixed:':
            kwargs['levels'] = self.ct_num_levels.value
        elif self.ct_num_levels_opts.value == 'option':
            if levels:
                if len(levels) > len(set(levels)):
                    if self.ct_auto_color.value == 'manual':
                        self._print_ct_info('Found duplicated levels, last appearance takes precedence')
                        tmp = dict(zip(levels, kwargs['colors']))  # use dict to get rid of duplicates
                        levels, kwargs['colors'] = list(tmp.keys()), list(tmp.values())
                    else:
                        levels = list(set(levels))  # use dict to get rid of duplicates
                kwargs['levels'] = sorted(levels)
                print(levels)
                if kwargs['colors'] is not None:
                    ii = sorted(range(len(levels)), key=lambda k: levels[k])
                    c = kwargs['colors']
                    kwargs['colors'] = list(c[i] for i in ii)
            else:
                self._print_ct_info('number of levels = option: but no option is added, fall back to auto')

        if self.ct_auto_color.value == 'colormap' and self.ct_num_levels_opts.value == 'auto' and self.ct_opts_list.children:
            self._print_ct_info('Level options have no effect.', 2)
        return kwargs

    def _ct_plot(self, pltfunc, if_clabel, if_inline_clabel, kwargs):
        if kwargs:  # this is called to replot for new data, kwargs contain all necessary info to generate the contour, kwargs will be altered
            pltfunc = kwargs.pop('method', 'contour')
            if_clabel = kwargs.pop('if_clabel', self.ct_if_clabel.value)
            if_inline_clabel = kwargs.pop('inline', self.ct_if_inline_clabel) and if_clabel
        else: # this is called when user press the plot button, kwargs will return keywords for the contour plot
            kwargs.update(self._extract_ct_kwargs_from_wgt())
        if pltfunc == 'contour':
            im2, _ = osh5vis.oscontour(self.__pp(self._data[self._slcs]), ax=self.ax, fig=self.fig, colorbar=False,
                                       xlabel=False, ylabel=False, title=False, **kwargs)
        else:
            im2, _ = osh5vis.oscontourf(self.__pp(self._data[self._slcs]), ax=self.ax, fig=self.fig, colorbar=False,
                                        xlabel=False, ylabel=False, title=False, **kwargs)
        kwargs['levels'] = im2.levels
        # have to set_alpha after the plot, otherwise alpha value cannot be updated later (not sure why)
        alpha = kwargs.get('alpha', 0.5)
        im2.set_alpha(alpha)
        kwargs['alpha'] = alpha
        if if_clabel:
            cl = self.ax.clabel(im2, im2.levels, use_clabeltext=True, inline=if_inline_clabel)
        else:
            cl = tuple()
        self.im2.append((im2, cl))
        return kwargs

    def _ct_clear_lvl_opts(self, *_):
        for opt in self.ct_opts_list.children:
            for w in opt.children:
                w.close()
            opt.close()
        self.ct_opts_list.children = tuple()

    def _ct_delete_plot(self, btn):
        wgt, kwargs, im = self.ct_plot_dict.pop(btn)
        self.ct_plot_dict.pop(wgt.children[1])  # remove the alpha key
        self.im2.remove(im)
        for c in im[0].collections:
            c.remove()
        for l in im[1]:
            l.remove()
        tmp = list(self.ct_wgt_list.children)
        tmp.remove(wgt)
        self.ct_wgt_list.children = tuple(tmp)
        for w in wgt.children:
            w.close()
        wgt.close()

    def _ct_set_alpha(self, change):
        kwargs, im = self.ct_plot_dict[self.ct_plot_dict[change['owner']]][-2:]
        im.set_alpha(change['new'])
        kwargs['alpha'] = change['new']

    def _add_contour_plot_wgt(self, kwargs):
        identifier = widgets.Label(value='Levels=' + str(kwargs['levels']) + '; ', style={'description_width': 'initial'}, layout={'width': 'initial'})
        alpha = widgets.BoundedFloatText(value=kwargs.get('alpha', 0.5), min=0., max=1., step=0.01, description='opacity:', continuous_update=False,
                                         style={'description_width': 'initial'}, layout={'width': '120px'})
        db = _get_delete_btn(kwargs['method'])
        db.on_click(self._ct_delete_plot)
        alpha.observe(self._ct_set_alpha, 'value')

        wgt = widgets.HBox([identifier, alpha, db],  layout=Layout(border='solid 1px'))
        self.ct_plot_dict[db] = [wgt, kwargs, self.im2[-1]]
        self.ct_plot_dict[alpha] = db
        self.ct_wgt_list.children += (wgt,)

    def _add_contour_plot(self, *_):
        kw = self._ct_plot(self.ct_method.value, self.ct_if_clabel.value, self.ct_if_inline_clabel.value, {})
        # save clabel kwargs for replotting
        kw['if_clabel'] = self.ct_if_clabel.value
        kw['inline'] = self.ct_if_inline_clabel.value and self.ct_if_clabel
        kw['method'] = self.ct_method.value
        # clear the level options in the output
        self._ct_clear_lvl_opts()
        # add a widgets to handle further interaction
        self._add_contour_plot_wgt(kw)

    def _ct_destroy_all(self):
        # close all widgets
        for wgt in self.ct_wgt_list.children:
            for w in wgt.children:
                w.close()
            wgt.close()
        self.ct_wgt_list.children, self.ct_plot_dict = (), {}
        # remove all contours
        for im in self.im2:
            for c in im[0].collections:
                c.remove()
            for l in im[1]:
                l.remove()
        self.im2 = []

    def _get_level_added(self):
        return [lvl_wgt.children[0].value for lvl_wgt in self.ct_opts_list.children]

    def _add_one_ct_lvl_opt(self, lvl, co):
        lvl, cp, db = self._get_contour_opt_wgt(lvl, co, '100px', '150px')
        widgets.jslink((lvl, 'disabled'), (self.ct_level, 'disabled'))
        widgets.jslink((cp, 'disabled'), (self.ct_colorpicker, 'disabled'))
        db.on_click(self._remove_ct_lvl_opt)
        lvl_wgt = widgets.HBox([lvl, cp, db], layout=Layout(border='solid 1px'))
        self.ct_opts_dict[db] = lvl_wgt
        self.ct_opts_list.children += (lvl_wgt,)

    def _add_contour_lvl_opts(self, *_):
        level_list = re.split('\s*,\s*|\s+|\s*;\s*', self.ct_level.value)
        for lvl in level_list:
            try:
                lv = float(lvl)
            except ValueError:
                continue
            finally:
                self._add_one_ct_lvl_opt(lv, self.ct_colorpicker.value)

    def __update_xanaopts(self, change):
        opts = self.__analysis_def[change['new']]
        optlist = [k for k in opts.keys()]
        self.xanaopts.options, self.xanaopts.value = optlist, optlist[0]

    def _get_xana_data_descr(self, start, end, opts, name, description_only=False):
        fn = self.__analysis_def[name][opts]
        s, e = self._data[self._slcs].axes[0][start] * self.yconv, self._data[self._slcs].axes[0][end] * self.yconv
        data = None if description_only else self.__pp(fn(self._data[self._slcs][start:end, :], 0))
        posstr = '%.2f~%.2f' % (s, e)
        tp = 'ix= ' + str(start) + ' ~ ' + str(end) + ' ' + opts + ' ' + name
        return data, posstr, tp

    def __add_xana(self, change):
        start, end = self._data[self._slcs].loc.label2int(0, self.anaxmin.value / self.yconv), self._data[self._slcs].loc.label2int(0, self.anaxmax.value / self.yconv)
        if start < end:
            data, posstr, tp = self._get_xana_data_descr(start, end, self.xanaopts.value, self.xananame.value)
            self.__add_xline(data, posstr, tp, 240)

    def __update_yanaopts(self, change):
        opts = self.__analysis_def[change['new']]
        optlist = [k for k in opts[1].keys()]
        self.yanaopts.options, self.yanaopts.value = optlist, optlist[0]

    def _get_yana_data_descr(self, start, end, opts, name, description_only=False):
        fn = self.__analysis_def[name][opts]
        s, e = self._data[self._slcs].axes[1][start] * self.xconv, self._data[self._slcs].axes[1][end] * self.xconv
        data = None if description_only else self.__pp(fn(self._data[self._slcs][:, start:end], 1))
        posstr = '%.2f~%.2f' % (s, e)
        tp = 'iy= ' + str(start) + ' ~ ' + str(end) + ' ' + opts + ' ' + name
        return data, posstr, tp

    def __add_yana(self, change):
        start, end = self._data[self._slcs].loc.label2int(1, self.anaymin.value), self._data[self._slcs].loc.label2int(1, self.anaymax.value)
        if start < end:
            data, posstr, tp = self._get_yana_data_descr(start, end, self.yanaopts.value, self.yananame.value)
            self.__add_yline(data, posstr, tp, 240)

    def __update_twinx_scale(self):
        if self.norm_selector.value[0] == LogNorm:
            self.axx.set_yscale('log')
        elif self.norm_selector.value[0] == SymLogNorm:
            self.axx.set_yscale('symlog')
        else:
            self.axx.set_yscale('linear')

    def __destroy_all_xlineout(self):
        if self._xlineouts:
            for li in self.xlineout_list_wgt.children:
                # remove lineout
                self._xlineouts[li.children[0]].remove()
                # remove widget
                li.close()
            # unregister all widgets
            self._xlineouts = {}
            self.xlineout_list_wgt.children = tuple()
            # remove axes
            self.axx.remove()

    def __remove_xlineout(self, btn):
        # unregister widget
        xlineout_wgt = self._xlineouts.pop(btn)
        xlineout = self._xlineouts.pop(xlineout_wgt.children[0])
        # remove x lineout
        xlineout.remove()
        # remove x lineout item widgets
        tmp = list(self.xlineout_list_wgt.children)
        tmp.remove(xlineout_wgt)
        self.xlineout_list_wgt.children = tuple(tmp)
        xlineout_wgt.close()
        # remove axes if all lineout is deleted
        if not self._xlineouts:
            self.axx.remove()
#         #TODO: a walkaround for a strange behavior of constrained_layout

    def __set_xlineout_color(self, color):
        self._xlineouts[color['owner']].set_color(color['new'])

    def __add_xline(self, data, descr, tp, wgt_width):
        # add twinx if not exist
        if not self._xlineouts:
            self.axx = self.ax.twinx()
            self.__update_twinx_scale()
        # plot
        xlineout = osh5vis.osplot1d(data, ax=self.axx, xlabel='', ylabel='', title='')[0]
        # add widgets (color picker + delete button)
        nw = _get_delete_btn(tp)
        nw.on_click(self.__remove_xlineout)
        co = xlineout.get_color()
        cpk = widgets.ColorPicker(concise=False, description=descr, value=co, style={'description_width': 'initial'},
                                  layout=Layout(width='%dpx' % wgt_width))
        cpk.observe(self.__set_xlineout_color, 'value')
        lineout_wgt = widgets.HBox([cpk, nw], layout=Layout(width='%dpx' % (wgt_width + 50), border='solid 1px', flex='0 0 auto'))
        self.xlineout_list_wgt.children += (lineout_wgt,)
        # register a new lineout
        self._xlineouts[nw], self._xlineouts[cpk] = lineout_wgt, xlineout

    def get_xlineout_data_and_index(self, loc):
        pos = self._data.loc.label2int(0, loc)
        posstr = '%.2f' % (self._data.axes[0][pos] * self.yconv)
        return self.__pp(self._data[pos, :][self._slcs[1]]), posstr, pos

    def __add_xlineout(self, *_):
        data, posstr, pos = self.get_xlineout_data_and_index(self.xlineout_wgt.value / self.yconv)
        tp = str(self._data.axes[0][pos]) + ' lineout'
        self.__add_xline(data, posstr, tp, 170)

    def __update_xlineout(self):
        if self._xlineouts:
#             for k, v in self._xlineouts.items():
#                 if hasattr(v, 'set_ydata'):
#                     v.set_ydata(self.__pp(v.get_ydata()))
#             for wgt in self.xlineout_list_wgt.children:
#                 pos = float(wgt.children[0].description)
#                 self._xlineouts[wgt.children[0]].set_ydata(self.__pp(self._data[self._slcs].loc[pos, :]))
            self.__update_twinx_scale()
            #TODO: autoscale for 'log' scale doesn't work after plotting the line, we have to do it manually
            #TDDO: a walkaround for a strange behavior of constrained_layout, should be removed in the future
            self.axx.set_ylabel('')

    def __update_twiny_scale(self):
        if self.norm_selector.value[0] == LogNorm:
            self.axy.set_xscale('log')
        elif self.norm_selector.value[0] == SymLogNorm:
            self.axy.set_xscale('symlog')
        else:
            self.axy.set_xscale('linear')

    def __destroy_all_ylineout(self):
        if self._ylineouts:
            for li in self.ylineout_list_wgt.children:
                # remove lineout
                self._ylineouts[li.children[0]].remove()
                # remove widget
                li.close()
            # unregister all widgets
            self._ylineouts = {}
            self.ylineout_list_wgt.children = tuple()
            # remove axes
            self.axy.remove()

    def __remove_ylineout(self, btn):
        # unregister widget
        ylineout_wgt = self._ylineouts.pop(btn)
        ylineout = self._ylineouts.pop(ylineout_wgt.children[0])
        # remove x lineout
        ylineout.remove()
        # remove x lineout item widgets
        tmp = list(self.ylineout_list_wgt.children)
        tmp.remove(ylineout_wgt)
        self.ylineout_list_wgt.children = tuple(tmp)
        ylineout_wgt.close()
        # remove axes if all lineout is deleted
        if not self._ylineouts:
            self.axy.remove()

    def __set_ylineout_color(self, color):
        self._ylineouts[color['owner']].set_color(color['new'])

    def __add_yline(self, data, descr, tp, wgt_width):
        # add twinx if not exist
        if not self._ylineouts:
            self.axy = self.ax.twiny()
            self.__update_twiny_scale()
        # plot
        ylineout = osh5vis.osplot1d(data, ax=self.axy, xlabel='', ylabel='', title='', transpose=True)[0]
        # add widgets (color picker + delete button)
        nw = _get_delete_btn(tp)
        nw.on_click(self.__remove_ylineout)
        co = ylineout.get_color()
        cpk = widgets.ColorPicker(concise=False, description=descr, value=co, style={'description_width': 'initial'},
                                  layout=Layout(width='%dpx' % wgt_width))
        cpk.observe(self.__set_ylineout_color, 'value')
        lineout_wgt = widgets.HBox([cpk, nw], layout=Layout(width='%dpx' % (wgt_width + 50), border='solid 1px', flex='0 0 auto'))
        self.ylineout_list_wgt.children += (lineout_wgt,)
        # register a new lineout
        self._ylineouts[nw], self._ylineouts[cpk] = lineout_wgt, ylineout

    def get_ylineout_data_and_index(self, loc):
        pos = self._data.loc.label2int(1, loc)
        posstr = '%.2f' % (self._data.axes[1][pos] * self.xconv)
        return self.__pp(self._data[:, pos][self._slcs[0]]), posstr, pos

    def __add_ylineout(self, *_):
        data, posstr, pos = self.get_ylineout_data_and_index(self.ylineout_wgt.value / self.xconv)
        tp = str(self._data.axes[1][pos]) + ' lineout'
        self.__add_yline(data, posstr, tp, 170)

    def __update_ylineout(self):
        if self._ylineouts:
#             for wgt in self.ylineout_list_wgt.children:
#                 pos = float(wgt.children[0].description)
#                 self._ylineouts[wgt.children[0]].set_xdata(self.__pp(self._data[self._slcs].loc[:, pos]))
            self.__update_twiny_scale()
            #TODO: autoscale for 'log' scale doesn't work after plotting the line, we have to do it manually
            #TDDO: a walkaround for a strange behavior of constrained_layout, should be removed in the future
            self.axy.set_ylabel('')

    def __handle_lognorm(self):
        s = self._data.shape
        dx, dy = 10 if s[1] > 200 else 1, 10 if s[0] > 200 else 1
        v = self._data.values[::dy, ::dx]
        if self.norm_selector.value[0] == LogNorm:
            self.__pp = np.abs
            self.vlogmin_wgt.value, self.vmax_wgt.value = np.min(np.abs(v[v!=0])), np.max(np.abs(v))
#             vmin, _ = self.__get_vminmax()
#             self.__assgin_valid_vmin(v=vmin)
        else:
            self.vmin_wgt.value, self.vmax_wgt.value = np.min(v), np.max(v)
#             self.__assgin_valid_vmin()
            self.__pp = do_nothing

    def __update_norm_wgt(self, change):
        """update tab1 (second tab) only and prepare _log_data if necessary"""
        tmp = [None] * len(self.tab_contents)
        tmp[0] = self.get_tab_data()
        self.refresh_tab_wgt(tmp)
        self.__handle_lognorm()
        self.__old_norm = change['old']

    def __get_vminmax(self, from_widgets=False):
        if from_widgets:
            return self.norm_selector.value[1].children[1].children[0].value, self.vmax_wgt.value
        else:
            return (None if self.if_vmin_auto.value else self.norm_selector.value[1].children[1].children[0].value,
                    None if self.if_vmax_auto.value else self.vmax_wgt.value)

    def __axis_descr_format(self, comp):
        return osh5vis.axis_format(self._data.axes[comp].long_name, self._data.axes[comp].units)

    def update_norm(self, *args):
        # only changing clim
        if self.__old_norm == self.norm_selector.value:
            vmin, vmax = self.__get_vminmax(from_widgets=True)
            self.im.set_clim([vmin, vmax])
        # norm change
        else:
            vminmax = self.__get_vminmax()
            self.__update_xlineout()
            self.__update_ylineout()
            self.im.remove()
            if self.colorbar.value:
                self.im, cb = self.plot_data()
                self.cb.ax.remove()
                self.cb = cb
            else:
                self.im, _ = self.plot_data(colorbar=False)

    def __get_norm(self, vminmax_from_wiget=False):
        vmin, vmax = self.__get_vminmax(vminmax_from_wiget)
        param = {'vmin': vmin, 'vmax': vmax, 'clip': self.if_clip_cm.value}
        if self.norm_selector.value[0] == PowerNorm:
            param['gamma'] = self.gamma.value
        if self.norm_selector.value[0] == SymLogNorm:
            param['linthresh'] = self.linthresh.value
            param['linscale'] = self.linscale.value
        return param

    def __assgin_valid_vmin(self, v=None):
        # if it is log scale
        if self.norm_selector.value[0] == LogNorm:
            self.vlogmin_wgt.value = self.eps if v is None or v < self.eps else v
        else:
            self.vmin_wgt.value = np.min(self.__pp(self._data[self._slcs])) if v is None else v

    def __add_colorbar(self):
        clb = self.cbar.value
        self.cb = osh5vis.add_colorbar(self.im, fig=self.fig, ax=self.ax, cblabel=clb)

    def __toggle_colorbar(self, change):
        if change['new']:
            self.cbar.disabled, self.if_reset_cbar.disabled, self.cmap_selector.disabled, \
            self.cmap_reverse.disabled = False, False, False, False
            self.__update_cbar(change)
            self.__add_colorbar()
        else:
            self.cbar.disabled, self.if_reset_cbar.disabled, self.cmap_selector.disabled, \
            self.cmap_reverse.disabled = True, True, True, True
            self.cb.remove()
#         self.replot_axes()

    def __update_vmin(self, _change):
        if self.if_vmin_auto.value:
            self.__assgin_valid_vmin()
            self.vmin_wgt.disabled = True
            self.vlogmin_wgt.disabled = True
        else:
            self.vmin_wgt.disabled = False
            self.vlogmin_wgt.disabled = False

    def __update_vmax(self, _change):
        if self.if_vmax_auto.value:
            self.vmax_wgt.value = np.max(self.__pp(self._data[self._slcs]))
            self.vmax_wgt.disabled = True
        else:
            self.vmax_wgt.disabled = False

    def __update_title(self, *_):
        if self.if_reset_title.value:
            self.datalabel.value = osh5vis.default_title(self._data, show_time=False)
            self.datalabel.disabled = True
        else:
            self.datalabel.disabled = False

    def __update_xlabel(self, change):
        if change['new']:
            self.xlabel.value = self._xlabel

    def __update_ylabel(self, change):
        if change['new']:
            self.ylabel.value = self._ylabel

    def __update_cbar(self, *_):
        if self.if_reset_cbar.value:
            self.cbar.value = self._data.units.tex()
            self.cbar.disabled = True
        else:
            self.cbar.disabled = False

    def __reset_save_button(self, *_):
        self.saveas.description, self.saveas.tooltip, self.saveas.button_style= \
        'Save current plot', 'save current plot', ''

    def __savefig(self):
        try:
            self.fig.savefig(self.figname.value, dpi=self.dpi.value)
#             self.dlink.clear_output(wait=True)
            self.dlink.value, self.dlink.description = _get_downloadable_url(self.figname.value), 'Download:'
            self.__reset_save_button(0)
        except PermissionError:
            self.saveas.description, self.saveas.tooltip, self.saveas.button_style= \
                    'Permission Denied', 'please try another directory', 'danger'

    def __try_savefig(self, *_):
        pdir = os.path.abspath(os.path.dirname(self.figname.value))
        path_exist = os.path.exists(pdir)
        file_exist = os.path.exists(self.figname.value)
        if path_exist:
            if file_exist:
                if not self.saveas.button_style:
                    self.saveas.description, self.saveas.tooltip, self.saveas.button_style= \
                    'Overwirte file', 'overwrite existing file', 'warning'
                else:
                    self.__savefig()
            else:
                self.__savefig()
        else:
            if not self.saveas.button_style:
                self.saveas.description, self.saveas.tooltip, self.saveas.button_style= \
                'Create path & save', 'create non-existing path and save', 'warning'
            else:
                os.makedirs(pdir)
                self.__savefig()


class Slicer(Generic2DPlotCtrl):
    def __init__(self, data, d=0, **extra_kwargs):
        if np.ndim(data) != 3:
            raise ValueError('data must be 3 dimensional')
        self.x, self.comp, self.data = data.shape[d] // 2, d, data
        self.slcs = self.__get_slice(d)
        self.axis_pos = widgets.FloatText(value=data.axes[self.comp][self.x],
                                          description=self.__axis_format(), continuous_update=False)
        self.index_slider = widgets.IntSlider(min=0, max=self.data.shape[self.comp] - 1, step=1, description='index:',
                                              value=self.data.shape[self.comp] // 2, continuous_update=False)

        self.axis_selector = widgets.Dropdown(options=list(range(data.ndim)), value=self.comp, description='axis:')
        self.if_pos_in_title = widgets.Checkbox(value=False, description='Slider position in title', layout=_items_layout)
        super(Slicer, self).__init__(data[tuple(self.slcs)], slcs=tuple(i for i in self.slcs if not isinstance(i, int)),
                                     time_in_title=not data.has_axis('t'), **extra_kwargs)
        self.axis_selector.observe(self.switch_slice_direction, 'value')
        self.index_slider.observe(self.update_slice, 'value')
        self.axis_pos.observe(self.__update_index_slider, 'value')
        self.if_pos_in_title.observe(self.update_title, 'value')
        #TODO: this is probably not a good design as it might break anytime we change the structure of tab0, refactory needed.
        # add the check box to tab0,
#         tmp = [None] * len(self.tab_contents)
#         tmp[0] = self.get_tab_data()
#         self.refresh_tab_wgt(tmp)

    def get_tab_data(self):
        return widgets.HBox([widgets.VBox([self.norm_selector, self.norm_selector.value[1]]), self.norm_btn_wgt,
                             widgets.VBox([widgets.HBox([self.datalabel, self.if_reset_title]),
                                           widgets.HBox([self.if_show_time, self.time_in_cgs]), self.if_pos_in_title])])
    @property
    def widgets_list(self):
        return self.tab, self.axis_pos, self.index_slider, self.axis_selector, self.out_main

    @property
    def widget(self):
        return widgets.VBox([widgets.HBox([self.axis_pos, self.index_slider, self.axis_selector]),
                             self.out_main])

    def __update_index_slider(self, _change):
        self.index_slider.value = round((self.axis_pos.value - self.data.axes[self.comp].min)
                                        / self.data.axes[self.comp].increment)

    def __axis_format(self):
        return osh5vis.axis_format(self.data.axes[self.comp].long_name, self.data.axes[self.comp].units)

    def __get_slice(self, c):
        slcs = [slice(None)] * self.data.ndim
        slcs[c] = self.data.shape[c] // 2
        return slcs

    def get_plot_title(self):
        l =  self.datalabel.value or ''
        t = self.get_time_label() if self.if_show_time.value else ''
        if self.if_pos_in_title.value:
            s = self.axis_pos.description.split()
            n, u = s[0], ' '.join(s[1:])
            pos = n + '=' + '{:.2f}'.format(self.axis_pos.value) + ' ' + u
        else:
            pos = ''
        title = l + (', ' + t) if l else t
        return title + ((', ' + pos) if pos else pos)

    def switch_slice_direction(self, change):
        self.slcs, self.comp, self.x = \
            self.__get_slice(change['new']), change['new'], self.data.shape[change['new']] // 2
        self.reset_slider_index()
        self.__update_axis_descr()
        self.update_data(self.data[tuple(self.slcs)], slcs=tuple(i for i in self.slcs if not isinstance(i, int)))
        self.reset_plot_area()
        self.replot_axes()

    def reset_slider_index(self):
        # stop the observe while updating values
        self.index_slider.unobserve(self.update_slice, 'value')
        self.index_slider.max = self.data.shape[self.comp] - 1
        self.__update_axis_value()
        self.index_slider.observe(self.update_slice, 'value')

    def __update_axis_value(self, *_):
        self.axis_pos.value = str(self.data.axes[self.comp][self.x])

    def __update_axis_descr(self, *_):
        self.axis_pos.description = self.__axis_format()

    def update_slice(self, index):
        self.x = index['new']
        self.__update_axis_value()
        self.slcs[self.comp] = self.x
        self.redraw(self.data[tuple(self.slcs)])
        if self.if_pos_in_title.value:
            self.update_title()


class SaveMovieManager(object):
    def __init__(self, fig, gen1frame, frame_range=None, frame_range_opts=None):
        self.fig, self.gen1frame, self.figdir, self.basename, self.frame_range_opts = fig, gen1frame, './', 'movie', frame_range_opts
        stdout = subprocess.check_output(['ffmpeg', '-encoders', '-v', 'quiet']).decode()
        #TODO: sane options for each encoder
        self.known_encoders = {'libx265': ["-crf", "18", "-preset", "slow"],
                               'hevc': ["-crf", "18", "-preset", "slow"],
                               'libx264': ["-tune", "stillimage", "-crf", "18", "-preset", "slow", "-pix_fmt", "yuv420p"],
                               'h264': ["-tune", "stillimage", "-crf", "18", "-preset", "slow", "-pix_fmt", "yuv420p"],
                               'mpeg4': ["-q:v", "2"], 'mpeg': [], 'flv': [], 'qtrle': [], 'rawvideo': []}
        for encoder in self.known_encoders.keys():
            if encoder in stdout:
                self.encoder, self.dflt_savename, self.dflt_btn_dscr, self.dflt_btn_tp = \
                encoder, 'movie.mp4', 'Generate movie', 'convert a series of plots into a movie'
                break
        else:
            self.encoder, self.dflt_savename, self.dflt_btn_dscr, self.dflt_btn_tp = \
            None, 'movie.tgz', 'Generate tar file', 'ffmpeg not found. A tgz file will be generated.'
        self.fps = widgets.BoundedIntText(value=30, min=1, max=300, description='FPS:', layout=_items_layout, style={'description_width': 'initial'})
        self.frame_range = widgets.SelectionRangeSlider(options=[0], description='movie range:',
                                                        style={'description_width': 'initial', 'min_width': '220px'})
        self.update_frame_range(frame_range)
        self.filename = widgets.Text(value=self.dflt_savename, description='Movie name:', placeholder='movie name')
        self.savebtn = widgets.Button(description=self.dflt_btn_dscr, tooltip=self.dflt_btn_tp, button_style='')
        self.deletetemp = widgets.Checkbox(value=True, description='Delete temporary figures',
                                           layout=_items_layout, style={'description_width': 'initial'})
        self.smm_output, self.dlink = Output(), widgets.HTML(value='', description='')
        self.whatif_file_exist = widgets.RadioButtons(options=['delete conflicting png files and continue', 'skip existing files',
                                                               'create new temporary directory'], value='skip existing files',
                                                      description='', layout=_items_layout, style={'description_width': 'initial'})
        # link widget events
        self.filename.observe(self.set_savename, 'value')
        self.savebtn.on_click(self.generate_figures)

    @property
    def widgets_list(self):
        return self.filename, self.frame_range, self.deletetemp, self.fps, self.savebtn, self.dlink, self.smm_output

    @property
    def widget(self):
        return widgets.VBox([widgets.HBox([self.filename, self.frame_range, self.deletetemp, self.fps, self.savebtn], layout=_items_layout),
                             self.dlink, self.smm_output], layout=_items_layout)

    def update_frame_range(self, frame_range):
        if frame_range is not None:
            starting, ending = frame_range if len(frame_range) == 2 else (0, frame_range[-1])
            self.frame_range_opts = range(starting, ending)
            self.frame_range.options = self.frame_range_opts
            self.frame_range.index = starting, ending - 1

    def set_savename(self, change):
        self.__reset_save_button()
#         self.smm_output.clear_output()
        self.basename = os.path.splitext(os.path.basename(change['new']))[0]

    def __reset_save_button(self, *_):
        self.savebtn.description, self.savebtn.tooltip, self.savebtn.button_style, self.savebtn.disabled = \
        self.dflt_btn_dscr, self.dflt_btn_tp, '', False

    def _delete_temp_fig_files(self):
        deleteme = glob.glob(self.figdir + '/' + self.basename + '-*.png')
        self.savebtn.description, self.savebtn.tooltip, self.savebtn.button_style, self.savebtn.disabled = \
        "Deleting", 'Unstopptable', 'info', True
        for f in deleteme:
            os.remove(f)

    def handle_path_file_conflict(self):
        """ what to do when user press the button. return True to indicate no further action required """
#         self.smm_output.clear_output()
        if self.savebtn.button_style == 'warning':  # second press, we have path/file problem to sort out
            if self.whatif_file_exist.index == 0:
                self._delete_temp_fig_files()
            elif self.whatif_file_exist.index == 2:
                self.figdir += datetime.now().strftime('__%Y-%m-%d_%H-%M-%S')
                os.makedirs(self.figdir)
        elif self.savebtn.button_style == 'info':  # second or third press, we are generating the movie, user call for abort
            self.__reset_save_button()
            return True
        elif not self.savebtn.button_style:  # user press the button for the first time
            self.savebtn.description, self.savebtn.tooltip, self.savebtn.button_style = \
            'continue', 'continue with selected option', 'warning'
            with self.smm_output:
                print('directory ' + self.figdir + ' not empty')
                display(self.whatif_file_exist)
            return True
        else:
            return True

    def __encode(self):
        if self.encoder:
            try:
                subprocess.check_output(["ffmpeg", "-framerate", str(self.fps.value),
                                         "-start_number", str(self.frame_range.index[0]),
                                         "-i", self.figdir + '/' + self.basename + '-%06d.png',
                                         '-c:v', self.encoder, *self.known_encoders[self.encoder],
                                         '-y', self.filename.value], stderr=subprocess.STDOUT)
            except subprocess.CalledProcessError as exc:
                self.__reset_save_button()
                with self.smm_output:
                    print(exc.output)
                return True  # pretty sure this is not how it should work
        else:
            with self.smm_output:
                subprocess.check_output(["tar", "-cvzf", self.filename.value, self.figdir + '/' + self.basename + '-*.png'])

    def _plot_and_encode(self):
        for i in range(self.frame_range.index[0], self.frame_range.index[1] + 1):
            self.__savefig(i)
            self.savebtn.description = "(%d/%d) done" % (i - self.frame_range.index[0] + 1, self.frame_range.index[1] - self.frame_range.index[0] + 1)
        else:
            self.savebtn.description = "Encoding"
            if self.__encode():
                return True
            if self.deletetemp.value:
                self._delete_temp_fig_files()
                try:
                    os.rmdir(self.figdir)
                except OSError:
                    pass
            self.dlink.value, self.dlink.description = _get_downloadable_url(self.filename.value), 'Download:'
        self.__reset_save_button()

    def generate_figures(self, *_):
        self.figdir = os.path.abspath(os.path.dirname(self.filename.value)) + '/' + self.basename
        try:
            if not os.path.exists(self.figdir):
                os.makedirs(self.figdir)
            # find out whether we are risking overwriting files
            if glob.glob(self.figdir + '/' + '*.png'):
                if self.handle_path_file_conflict():
                    return
            self.savebtn.description, self.savebtn.tooltip, self.savebtn.button_style, self.savebtn.disabled = \
            "Generating Frames", 'Use menu "Kernel --> interrput" to stop', 'info', True
            #TODO: matplotlib is not thread safe so we have to plot in the main thread (need further investigation)
            self._plot_and_encode()
            with self.smm_output:
                print(worker.is_alive())

        except PermissionError:
            self.savebtn.description, self.savebtn.tooltip, self.savebtn.button_style = \
                        'Error', 'please try another path', 'danger'
            with self.smm_output:
                print('Permission Denied: cannot write to ' + self.figdir)
        except KeyboardInterrupt:
            self.__reset_save_button()

    def __savefig(self, i):
        savename = self.figdir + '/' + self.basename + '-{:06d}'.format(i) + '.png'
        if os.path.exists(savename) and self.whatif_file_exist.index == 1:
            return
        self.gen1frame(i)
        self.fig.savefig(savename, dpi=300)


class DirSlicer(Generic2DPlotCtrl):
    def __init__(self, filefilter, processing=do_nothing, savemovie=None, **extra_kwargs):
        if isinstance(filefilter, (tuple, list)):  # filefilter is a list of filenames
            self.datadir, self.flist, self.processing = os.path.abspath(os.path.dirname(filefilter[0])), filefilter, processing
        else:  # filefilter is a path name or a string that can be passed down to glob to get a list of file names
            fp = filefilter + '/*.h5' if os.path.isdir(filefilter) else filefilter
            self.datadir, self.flist, self.processing = os.path.abspath(os.path.dirname(fp)), sorted(glob.glob(fp)), processing
        try:
            self.data = processing(osh5io.read_h5(self.flist[0]))
        except IndexError:
            raise IOError('No file found matching ' + fp)

        self.file_slider = widgets.IntSlider(min=0, max=len(self.flist) - 1, description=os.path.basename(self.flist[0]),
                                             value=0, continuous_update=False, layout=_items_layout,
                                             style={'description_width': 'initial'})
        self.time_label = widgets.Label(value=osh5vis.time_format(self.data.run_attrs['TIME'][0], self.data.run_attrs['TIME UNITS']),
                                        layout=_items_layout)

        super(DirSlicer, self).__init__(self.data, time_in_title=False, **extra_kwargs)
        if savemovie is None:
            self.savemovie = SaveMovieManager(self.fig, self.plot_ith_slice, (0, len(self.flist)))
        else:
            self.savemovie = savemovie
            self.savemovie.update_frame_range((0, len(self.flist)))
        tmp = [None] * len(self.tab_contents)
        tmp[4] = self.__get_tab_save()
        self.refresh_tab_wgt(tmp)
        self.file_slider.observe(self.update_slice, 'value')

    @property
    def widgets_list(self):
        return self.tab, self.file_slider, self.time_label, self.out_main

    @property
    def widget(self):
        return widgets.VBox([widgets.HBox[self.file_slider, self.time_label], self.out_main])

    # cannot overload get_tab_save() because of circular dependent between SaveMovieManager __init__ and Generic2DPlotCtrl __init__
    def __get_tab_save(self):
        return widgets.VBox([widgets.HBox([self.figname, self.dpi, self.saveas], layout=_items_layout),
                             self.dlink, self.savemovie.widget], layout=_items_layout)

    def plot_ith_slice(self, i):
        c = {'new': i}
        self.update_slice(c)

    def update_contours(self):
        for wgt in self.ct_wgt_list.children:
            _, _, db = wgt.children
            w, kwargs, im = self.ct_plot_dict[db]
            # reconstruct the contour
            self.im2.remove(im)
            # free up resources
            for c in im[0].collections:
                c.remove()
            for l in im[1]:
                l.remove()
            # _ct_plot will append im2 with the new plot
            tmp = kwargs.copy()
            self._ct_plot(None, None, None, tmp)
            self.ct_plot_dict[db][-1] = self.im2[-1]

    def update_slice(self, change):
        self.file_slider.description = os.path.basename(self.flist[change['new']])
        self.data = self.processing(osh5io.read_h5(self.flist[change['new']]))
        self.time_label.value = osh5vis.time_format(self.data.run_attrs['TIME'][0], self.data.run_attrs['TIME UNITS'])
        self.redraw(self.data, update_vminmax=True)
        self.update_lineouts()
        self.update_contours()
        if self.if_show_time.value:
#             self.update_time_label()
            self.update_title(change)

    def select_ith_file(self, i):
        self.file_slider.value = i


class MultiPanelCtrl(object):
    def __init__(self, workers, data_list, grid, worker_kw_list=None, figsize=None, fig_ax=tuple(), output_widget=None,
                 sharex=False, sharey=False, **kwargs):
        """ worker's base class should be Generic2DPlotCtrl """
        if len(grid) != 2 or np.multiply(*grid) <= 1:
            raise ValueError('grid must have 2 elements specifying a grid of plots. Total number of plots must be greater than 1')
        self.nrows, self.ncols = grid
        if len(data_list) != self.nrows * self.ncols:
            raise ValueError('Expecting %d lists in data_list, got %d' % (self.nrows * self.ncols, len(data_list)))

        width, height = figsize or plt.rcParams.get('figure.figsize')
        self.out = output_widget or widgets.Output()
        nplots = self.nrows * self.ncols
        xlabel, ylabel = [None,] * nplots, [None,] * nplots
        if str(sharex).lower() in ('true', 'all', 'col'):
            for i in range(nplots - self.ncols):
                xlabel[i] = False
        if str(sharey).lower() in ('true', 'all', 'row'):
            for i in range(nplots):
                if i % self.ncols != 0:
                    ylabel[i] = False
        if worker_kw_list is None:
            worker_kw_list = ({}, ) * nplots
        self.fig, axes = fig_ax or plt.subplots(self.nrows, self.ncols, figsize=(width, height), sharex=sharex, sharey=sharey, constrained_layout=True)
        self.ax = axes.flatten()
        dstr = kwargs.pop('onDestruction', self.self_destruct)
        self.worker = [w(d, output_widget=self.out, fig=self.fig, ax=ax, xlabel=xlb, ylabel=ylb, onDestruction=dstr, **wkw, **kwargs)
                       for w, d, ax, xlb, ylb, wkw in zip(workers, data_list, self.ax, xlabel, ylabel, worker_kw_list)]
        data_namelist = [s.get_dataname() for s in self.worker]
        # adding the index in front to make sure all button names are unique (otherwise the selection wouldn't be highlighted properly)
        if len(data_namelist) > len(set(data_namelist)):
            data_namelist = [str(i+1)+'.'+s for i, s in enumerate(data_namelist)]
        self.tabd = [s.tab for s in self.worker]
        bw, bwpadded = 50, 56  # these magic numbers seems to work well on forcing the desired button layout
        self.tb = widgets.ToggleButtons(options=data_namelist, value=data_namelist[0], description='', tooltips=data_namelist,
                                        style={"button_width": '%dpx' % bw})
        ctrl_pnl = widgets.Box([self.tb],layout=Layout(display='flex', flex='0 0 auto', align_items='center',
                                                       width='%dpx' % (bwpadded * self.ncols)))
        self.ctrl = widgets.HBox([ctrl_pnl, self.tabd[self.tb.index]], layout=_items_layout)
        self.suptitle_wgt = widgets.Text(value=None, placeholder='None', continuous_update=False, description='Suptitle:')
        self.time_in_suptitle = widgets.Checkbox(value=False, description='Time in suptitle, ', style={'description_width': 'initial'})
        self.time_in_cgs_unit = widgets.Checkbox(value=False, description='time in cgs unit', style={'description_width': 'initial'})
        self.suptitle = widgets.HBox([self.suptitle_wgt, self.time_in_suptitle, self.time_in_cgs_unit])
        #TODO: some axes setting should be shared (using widgets.jslink maybe) among different panels
        # disable resize widgets to avoid bugs
        if sharex or sharey:
            for s in self.worker:
                s.x_min_wgt.disabled, s.y_min_wgt.disabled, s.x_max_wgt.disabled, s.y_max_wgt.disabled, \
                s.x_step_wgt.disabled, s.y_step_wgt.disabled = (True,) * 6
        # link widget events
        self.tb.observe(self.show_corresponding_tab, 'index')
        self.suptitle_wgt.observe(self.update_suptitle, 'value')
        self.time_in_suptitle.observe(self.update_suptitle, 'value')
        self.time_in_cgs_unit.observe(self.update_suptitle, 'value')

    @property
    def widgets_list(self):
        return self.ctrl, self.suptitle, self.out

    @property
    def time(self):
        return self.worker[0].get_time_label(convert_tunit=self.time_in_cgs_unit.value)

    def self_destruct(self):
        self.ctrl.close()
        self.suptitle.close()

    def update_suptitle(self, *_):
        if self.suptitle_wgt.value:
            ttl = self.suptitle_wgt.value + ((', ' + self.time) if self.time_in_suptitle.value else '')
        else:
            ttl = self.time if self.time_in_suptitle.value else None
        if ttl is not None:
            self.fig.suptitle(ttl)

    def show_corresponding_tab(self, change):
        self.ctrl.children = (self.ctrl.children[0], self.tabd[self.tb.index])


class MPDirSlicer(MultiPanelCtrl):
    def __init__(self, filefilter_list, grid, interval=1000, processing=do_nothing, figsize=None, fig_ax=tuple(), output_widget=None,
                 sharex=False, sharey=False, worker_kw_list=None, **kwargs):
        if worker_kw_list is None:
            worker_kw_list = [{},] * (grid[0] * grid[1])
        if isinstance(processing, (list, tuple)):
            if len(processing) != grid[0] * grid[1]:
                raise ValueError('Expecting %d functions in processing, got %d' % (grid[0] * grid[1], len(processing)))
            else:
                ps = [{'processing' :p} for p in processing]
        else:
            ps = ({'processing' :processing},) * len(filefilter_list)
        for kw, p in zip(worker_kw_list, ps):
            kw.update(p)
        #TODO: replicating some lines in the super class. there should be a better way >>>>>
        width, height = figsize or plt.rcParams.get('figure.figsize')
        self.fig, self.ax = fig_ax or plt.subplots(*grid, figsize=(width, height), sharex=sharex, sharey=sharey, constrained_layout=True)
#         fp = filefilter_list[0] + '/*.h5' if os.path.isdir(filefilter_list[0]) else filefilter_list[0]
#         flist = sorted(glob.glob(fp))
        # <<<<< all because we need to initialize SaveMovieManager before super().__init__()
        smm = SaveMovieManager(self.fig, self.plot_ith_slice_mp)
        super(MPDirSlicer, self).__init__((DirSlicer,) * len(filefilter_list), filefilter_list, grid, worker_kw_list=worker_kw_list,
                                          figsize=figsize, fig_ax=(self.fig, self.ax), output_widget=output_widget, sharex=sharex,
                                          sharey=sharey, onDestruction=self.self_destruct, savemovie=smm, **kwargs)
        # we need a master slider to control all subplot sliders
        self.slider = widgets.IntSlider(min=0, max=self.worker[0].file_slider.max, description='', value=0,
                                        continuous_update=False, style={'description_width': 'initial'})
        self.play = widgets.Play(interval=interval, value=0, min=0, max=self.slider.max, description='Press play')
        self.slider.observe(self.update_all_subplots, 'value')
        widgets.jslink((self.play, 'value'), (self.slider, 'value'))

    @property
    def widgets_list(self):
        return self.ctrl, self.play, self.slider, self.worker[0].time_label, self.suptitle, self.out

    def self_destruct(self):
        try:
            self.worker[0].time_label.close()
        except IndexError:
            pass
        self.play.close()
        self.slider.close()
        super().self_destruct()

    def plot_ith_slice_mp(self, i):
        c = {'new': i}
        self.update_all_subplots(c)

    def update_all_subplots(self, change):
        for s in self.worker:
            s.file_slider.value = change['new']
        self.update_suptitle(change)


class Animation(Slicer):
    def __init__(self, data, interval=10, step=1, **kwargs):
        super(Animation, self).__init__(data, **kwargs)
        self.play = widgets.Play(interval=interval, value=self.x, min=0, max=len(self.data.axes[self.comp]),
                                 step=step, description="Press play", disabled=False)
        self.interval_wgt = widgets.IntText(value=interval, description='Interval:', disabled=False)
        self.step_wgt = widgets.IntText(value=step, description='Step:', disabled=False)

        # link everything together
        widgets.jslink((self.play, 'value'), (self.index_slider, 'value'))
        self.interval_wgt.observe(self.update_interval, 'value')
        self.step_wgt.observe(self.update_step, 'value')

    @property
    def widgets_list(self):
        return (self.tab, self.axis_pos, self.index_slider, self.axis_selector,
                self.play, self.interval_wgt, self.step_wgt, self.out_main)

    def switch_slice_direction(self, change):
        super(Animation, self).switch_slice_direction(change)
        self.play.max = len(self.data.axes[self.comp])

    def update_interval(self, change):
        self.play.interval = change['new']

    def update_step(self, change):
        self.play.step = change['new']
