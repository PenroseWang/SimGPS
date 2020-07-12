"""
Signal characteristics animation
"""

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import gridspec
matplotlib.use('TkAgg')


class sigCharacterAnimation:
    '''
    Animation for signal characteristics
    '''
    # signal attributes
    am1, am2 = None, None           # signal amplitude, should <=1
    f1, f2 = None, None             # signal frequency, should >=1, multiple of 1
    phase1, phase2 = None, None     # signal phase, [0, 2pi)
    t = None
    sin1, cos1 = None, None         # signal 1
    sin2, cos2 = None, None         # signal 2
    # animation attributes
    fig, axes = None, None
    lines, point = None, None
    frames = None
    transFigure = None

    def __init__(self, signal1=None, signal2=None):
        if signal1 is None:
            signal1 = [1, 1, 0]
        if signal2 is None:
            signal2 = [1, 2, np.pi / 2]
        self.am1, self.am2 = signal1[0], signal2[0]
        self.f1, self.f2 = signal1[1], signal2[1]
        self.phase1, self.phase2 = signal1[2], signal2[2]
        # generate signal
        min_freq = min(self.f1, self.f2)
        fs = 64 * min_freq
        self.t = np.arange(0, 1 / min_freq, 1 / fs)
        self.sin1 = self.am1 * np.sin(2 * np.pi * self.f1 * self.t + self.phase1)
        self.cos1 = self.am1 * np.cos(2 * np.pi * self.f1 * self.t + self.phase1)
        self.sin2 = self.am2 * np.sin(2 * np.pi * self.f2 * self.t + self.phase2)
        self.cos2 = self.am2 * np.cos(2 * np.pi * self.f2 * self.t + self.phase2)
        self.frames = len(self.t)

    def start(self, save_gif=False, f_name='correlation'):
        '''start animation or save as gif'''
        self._init_animation()
        anim = FuncAnimation(self.fig, self._animate, frames=self.frames, blit=False,
                             interval=100, init_func=self._init_canvas)
        if save_gif:
            if not os.path.exists('vis'):
                os.makedirs('vis')
            anim.save(os.path.join('vis', f_name + '.gif'), codec='png', writer='imagemagick')
        else:
            plt.show()

    def _init_animation(self):
        ''' initialize animation'''

        self.fig = plt.figure(figsize=(8, 6))
        self.axes = [matplotlib.axes.Axes] * 4
        gs = gridspec.GridSpec(2, 2, width_ratios=[1, 2])
        self.axes[0] = plt.subplot(gs[0, 0], aspect='equal')
        self.axes[1] = plt.subplot(gs[0, 1])
        self.axes[2] = plt.subplot(gs[1, 0], aspect='equal')
        self.axes[3] = plt.subplot(gs[1, 1])

        self.lines = [matplotlib.axes.Axes.plot] * 4
        self.points = [matplotlib.axes.Axes.plot] * 4
        self.connect = [matplotlib.lines.Line2D] * 2
        circ1 = plt.Circle((0, 0), self.am1, fill=False, linewidth=2, color='orange')
        circ2 = plt.Circle((0, 0), self.am2, fill=False, linewidth=2, color='green')
        self.axes[0].add_artist(circ1)
        self.axes[0].set_xlim(-1.1, 1.1)
        self.axes[0].set_ylim(-1.1, 1.1)
        self.axes[0].axis('off')
        self.axes[1].set_xlim(-0.1, 1)
        self.axes[1].set_ylim(-1.1, 1.1)
        self.axes[1].axis('off')
        self.axes[2].add_artist(circ2)
        self.axes[2].set_xlim(-1.1, 1.1)
        self.axes[2].set_ylim(-1.1, 1.1)
        self.axes[2].axis('off')
        self.axes[3].set_xlim(-0.1, 1)
        self.axes[3].set_ylim(-1.1, 1.1)
        self.axes[3].axis('off')

        self.lines[0], = self.axes[0].plot([], [], linewidth=2)
        self.lines[1], = self.axes[1].plot([], [], linewidth=2, color='orange')
        self.lines[2], = self.axes[2].plot([], [], linewidth=2)
        self.lines[3], = self.axes[3].plot([], [], linewidth=2, color='green')
        self.points[0], = self.axes[0].plot([], [], 'ro', markersize=6)
        self.points[1], = self.axes[1].plot([], [], 'ro', markersize=6)
        self.points[2], = self.axes[2].plot([], [], 'ro', markersize=6)
        self.points[3], = self.axes[3].plot([], [], 'ro', markersize=6)
        self.connect[0] = matplotlib.lines.Line2D([], [], color='r', transform=self.fig.transFigure)
        self.connect[1] = matplotlib.lines.Line2D([], [], color='r', transform=self.fig.transFigure)
        self.fig.lines.extend(self.connect)
        self.transFigure = self.fig.transFigure.inverted()

    def _init_canvas(self):
        '''do nothing, return artists to be re-drawn'''
        return self.lines + self.points + self.connect

    def _animate(self, i):
        '''perform animation step'''
        # update artists
        self.lines[0].set_data([0, self.cos1[-i]], [0, self.sin1[-i]])
        self.lines[1].set_data(self.t, np.roll(self.sin1, i))
        self.lines[2].set_data([0, self.cos2[-i]], [0, self.sin2[-i]])
        self.lines[3].set_data(self.t, np.roll(self.sin2, i))
        self.points[0].set_data([self.cos1[-i]], [self.sin1[-i]])
        self.points[1].set_data([0], [self.sin1[-i]])
        self.points[2].set_data([self.cos2[-i]], [self.sin2[-i]])
        self.points[3].set_data([0], [self.sin2[-i]])

        coord1 = self.transFigure.transform(self.axes[0].transData.transform([self.cos1[-i], self.sin1[-i]]))
        coord2 = self.transFigure.transform(self.axes[1].transData.transform([0, self.sin1[-i]]))
        self.connect[0].set_data((coord1[0], coord2[0]), (coord1[1], coord2[1]))
        coord1 = self.transFigure.transform(self.axes[2].transData.transform([self.cos2[-i], self.sin2[-i]]))
        coord2 = self.transFigure.transform(self.axes[3].transData.transform([0, self.sin2[-i]]))
        self.connect[1].set_data((coord1[0], coord2[0]), (coord1[1], coord2[1]))
        return self.lines + self.points + self.connect


if __name__ == '__main__':

    save_fig = True
    # amplitude
    # anim = sigCharacterAnimation(signal1=[1, 1, 0], signal2=[0.5, 1, 0])
    # anim.start(save_gif=save_fig, f_name='sig_char_amplitude')

    # # frequency
    # anim = sigCharacterAnimation(signal1=[1, 1, 0], signal2=[1, 2, 0])
    # anim.start(save_gif=save_fig, f_name='sig_char_frequency')

    # phase
    anim = sigCharacterAnimation(signal1=[1, 1, 0], signal2=[1, 1, np.pi / 2])
    anim.start(save_gif=save_fig, f_name='sig_char_phase')


