import numpy as np
import numpy.matlib
import scipy
import pandas as pd
from fractions import Fraction
import matplotlib as mpl
import matplotlib.collections as mcoll
import matplotlib.path as mpath
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
from matplotlib import cm, colors, gridspec
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import figurefirst as fifi
import fly_plot_lib_plot as fpl
# import utils


class LatexStates:
    """Holds LaTex format corresponding to set symbolic variables.
    """

    def __init__(self):
        self.dict = {'v_para': r'$v_{\parallel}$',
                     'v_perp': r'$v_{\perp}$',
                     'phi': r'$\phi$',
                     'phidot': r'$\dot{\phi}$',
                     'phi_dot': r'$\dot{\phi}$',
                     'phiddot': r'$\ddot{\phi}$',
                     'w': r'$w$',
                     'zeta': r'$\zeta$',
                     'w_dot': r'$\dot{w}$',
                     'zeta_dot': r'$\dot{\zeta}$',
                     'I': r'$I$',
                     'm': r'$m$',
                     'C_para': r'$C_{\parallel}$',
                     'C_perp': r'$C_{\perp}$',
                     'C_phi': r'$C_{\phi}$',
                     'km1': r'$k_{m_1}$',
                     'km2': r'$k_{m_2}$',
                     'km3': r'$k_{m_3}$',
                     'km4': r'$k_{m_4}$',
                     'd': r'$d$',
                     'psi': r'$\psi$',
                     'gamma': r'$\gamma$',
                     'alpha': r'$\alpha$',
                     'of': r'$\frac{g}{d}$',
                     'gdot': r'$\dot{g}$',
                     'v_para_dot': r'$\dot{v_{\parallel}}$',
                     'v_perp_dot': r'$\dot{v_{\perp}}$',
                     'v_para_dot_ratio': r'$\frac{\Delta v_{\parallel}}{v_{\parallel}}$',
                     'x':  r'$x$',
                     'y':  r'$y$',
                     'v_x': r'$v_{x}$',
                     'v_y': r'$v_{y}$',
                     'v_z': r'$v_{z}$',
                     'w_x': r'$w_{x}$',
                     'w_y': r'$w_{y}$',
                     'w_z': r'$w_{z}$',
                     'a_x': r'$a_{x}$',
                     'a_y': r'$a_{y}$',
                     'vx': r'$v_x$',
                     'vy': r'$v_y$',
                     'vz': r'$v_z$',
                     'wx': r'$w_x$',
                     'wy': r'$w_y$',
                     'wz': r'$w_z$',
                     'ax': r'$ax$',
                     'ay': r'$ay$',
                     'thetadot': r'$\dot{\theta}$',
                     'theta_dot': r'$\dot{\theta}$',
                     'psidot': r'$\dot{\psi}$',
                     'psi_dot': r'$\dot{\psi}$',
                     'theta': r'$\theta$',
                     'Yaw': r'$\psi$',
                     'R': r'$\phi$',
                     'P': r'$\theta$',
                     'dYaw': r'$\dot{\psi}$',
                     'dP': r'$\dot{\theta}$',
                     'dR': r'$\dot{\phi}$',
                     'acc_x': r'$\dot{v}x$',
                     'acc_y': r'$\dot{v}y$',
                     'acc_z': r'$\dot{v}z$',
                     'Psi': r'$\Psi$',
                     'Ix': r'$I_x$',
                     'Iy': r'$I_y$',
                     'Iz': r'$I_z$',
                     'Jr': r'$J_r$',
                     'Dl': r'$D_l$',
                     'Dr': r'$D_r$',
                     }

    def convert_to_latex(self, list_of_strings, remove_dollar_signs=False):
        """ Loop through list of strings and if any match the dict, then swap in LaTex symbol.
        """

        if isinstance(list_of_strings, str):  # if single string is given instead of list
            list_of_strings = [list_of_strings]
            string_flag = True
        else:
            string_flag = False

        list_of_strings = list_of_strings.copy()
        for n, s in enumerate(list_of_strings):  # each string in list
            for k in self.dict.keys():  # check each key in Latex dict
                if s == k:  # string contains key
                    list_of_strings[n] = self.dict[k]  # replace string with LaTex
                    if remove_dollar_signs:
                        list_of_strings[n] = list_of_strings[n].replace('$', '')

        if string_flag:
            list_of_strings = list_of_strings[0]

        return list_of_strings


def plot_trajectory(xpos, ypos, phi, color, ax=None, size_radius=None, nskip=0,
                    colormap=None, colornorm=None, edgecolor='none', reverse=False):
    if color is None:
        color = phi

    color = np.array(color)

    # Set size radius
    xymean = np.mean(np.abs(np.hstack((xpos, ypos))))
    if size_radius is None:  # auto set
        xymean = 0.21 * xymean
        if xymean < 0.0001:
            sz = np.array(0.01)
        else:
            sz = np.hstack((xymean, 1))
        size_radius = sz[sz > 0][0]
    else:
        if isinstance(size_radius, list):  # scale defualt by scalar in list
            xymean = size_radius[0] * xymean
            sz = np.hstack((xymean, 1))
            size_radius = sz[sz > 0][0]
        else:  # use directly
            size_radius = size_radius

    if colornorm is None:
        colornorm = [np.min(color), np.max(color)]

    if reverse:
        xpos = np.flip(xpos, axis=0)
        ypos = np.flip(ypos, axis=0)
        phi = np.flip(phi, axis=0)
        color = np.flip(color, axis=0)

    if colormap is None:
        colormap = cm.get_cmap('bone_r')
        colormap = colormap(np.linspace(0.1, 1, 10000))
        colormap = ListedColormap(colormap)

    if ax is None:
        fig, ax = plt.subplots()

    fpl.colorline_with_heading(ax, np.flip(xpos), np.flip(ypos), np.flip(color, axis=0), np.flip(phi),
                               nskip=nskip,
                               size_radius=size_radius,
                               deg=False,
                               colormap=colormap,
                               center_point_size=0.0001,
                               colornorm=colornorm,
                               show_centers=False,
                               size_angle=20,
                               alpha=1,
                               edgecolor=edgecolor)

    ax.set_aspect('equal')
    xrange = xpos.max() - xpos.min()
    xrange = np.max([xrange, 0.02])
    yrange = ypos.max() - ypos.min()
    yrange = np.max([yrange, 0.02])

    if yrange < (size_radius / 2):
        yrange = 10

    if xrange < (size_radius / 2):
        xrange = 10

    ax.set_xlim(xpos.min() - 0.2 * xrange, xpos.max() + 0.2 * xrange)
    ax.set_ylim(ypos.min() - 0.2 * yrange, ypos.max() + 0.2 * yrange)

    # fifi.mpl_functions.adjust_spines(ax, [])


def pi_yaxis(ax=0.5, tickpispace=0.5, lim=None, real_lim=None):
    if lim is None:
        ax.set_ylim(-1 * np.pi, 1 * np.pi)
    else:
        ax.set_ylim(lim)

    lim = ax.get_ylim()
    ticks = np.arange(lim[0], lim[1] + 0.01, tickpispace * np.pi)
    tickpi = np.round(ticks / np.pi, 3)
    y0 = abs(tickpi) < np.finfo(float).eps  # find 0 entry, if present

    tickslabels = tickpi.tolist()
    for y in range(len(tickslabels)):
        tickslabels[y] = ('$' + str(Fraction(tickslabels[y])) + r'\pi $')

    tickslabels = np.asarray(tickslabels, dtype=object)
    tickslabels[y0] = '0'  # replace 0 entry with 0 (instead of 0*pi)
    ax.set_yticks(ticks)
    ax.set_yticklabels(tickslabels)

    if real_lim is None:
        real_lim = np.zeros(2)
        real_lim[0] = lim[0] - 0.4
        real_lim[1] = lim[1] + 0.4

    if lim is None:
        ax.set_ylim(-1 * np.pi, 1 * np.pi)
    else:
        ax.set_ylim(lim)

    ax.set_ylim(real_lim)


def pi_xaxis(ax, tickpispace=0.5, lim=None):
    if lim is None:
        ax.set_xlim(-1 * np.pi, 1 * np.pi)
    else:
        ax.set_xlim(lim)

    lim = ax.get_xlim()
    ticks = np.arange(lim[0], lim[1] + 0.01, tickpispace * np.pi)
    tickpi = ticks / np.pi
    x0 = abs(tickpi) < np.finfo(float).eps  # find 0 entry, if present

    tickslabels = tickpi.tolist()
    for x in range(len(tickslabels)):
        tickslabels[x] = ('$' + str(Fraction(tickslabels[x])) + r'\pi$')

    tickslabels = np.asarray(tickslabels, dtype=object)
    tickslabels[x0] = '0'  # replace 0 entry with 0 (instead of 0*pi)
    ax.set_xticks(ticks)
    ax.set_xticklabels(tickslabels)


def circplot(t, phi, jump=np.pi):
    """ Stitches t and phi to make unwrapped circular plot. """

    t = np.squeeze(t)
    phi = np.squeeze(phi)

    difference = np.abs(np.diff(phi, prepend=phi[0]))
    ind = np.squeeze(np.array(np.where(difference > jump)))

    phi_stiched = np.copy(phi)
    t_stiched = np.copy(t)
    for i in range(phi.size):
        if np.isin(i, ind):
            phi_stiched = np.concatenate((phi_stiched[0:i], [np.nan], phi_stiched[i + 1:None]))
            t_stiched = np.concatenate((t_stiched[0:i], [np.nan], t_stiched[i + 1:None]))

    return t_stiched, phi_stiched


def colorline(x, y, z, ax=None, cmap=plt.get_cmap('copper'), norm=plt.Normalize(0.0, 1.0), linewidth=1.5, alpha=1.0):
    # Special case if a single number:
    if not hasattr(z, "__iter__"):  # to check for numerical input -- this is a hack
        z = np.array([z])

    z = np.asarray(z)

    segments = make_segments(x, y)
    lc = mcoll.LineCollection(segments, array=z, cmap=cmap, norm=norm,
                              linewidth=linewidth, alpha=alpha,
                              path_effects=[path_effects.Stroke(capstyle="round")])

    if ax is None:
        ax = plt.gca()

    ax.add_collection(lc)

    return lc


def make_segments(x, y):
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments
