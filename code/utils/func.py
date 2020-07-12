"""
Utilities for DSP functions
"""

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
matplotlib.use('TkAgg')
language = 'en'  # en: English or cn: Chinese
if language == 'cn':
    matplotlib.rcParams['font.family'] = ['Heiti TC']


# circular correlation
def circular_correlation(data1, data2):
    return np.correlate(data1, np.hstack((data2[1:], data2)), mode='valid') / len(data1)


# calculate phase differences of two signals with similar frequency
# both signals should be only one period
def phase_diff(data1, data2):

    len_data = len(data1)
    cross_corr = circular_correlation(data1, data2)
    max_idx = np.argmax(cross_corr)
    phase_diff = min(max_idx, len_data - max_idx) / len_data
    return phase_diff / np.pi * 360     # convert to degree


# calculate phase differences of two signals with similar frequency
# insensitive to bit modulation
def phase_diff_insensitive(data1, data2):

    phase_diff1 = phase_diff(data1, data2)
    phase_diff2 = phase_diff(-data1, data2)
    return min(phase_diff1, phase_diff2)


# add noise to signal based on SNR
def awgn(signal, snr):
    # signal power
    p_signal = 10 * np.log10(np.mean(signal ** 2))  # in dB
    p_noise = p_signal - snr
    sigma = np.sqrt(10 ** (p_noise / 10))
    noise = np.random.normal(0, sigma, len(signal))
    return signal + noise


class correlationAnimation:
    '''
    Animation for signal correlation
    '''
    signal1, signal2 = None, None   # original signals
    len1, len2 = None, None         # original signal length
    sig1, sig2 = None, None         # reconstructed signals
    normalized = None               # normalized or un-normalized correlation
    fig, axes = None, None
    lines, point = None, None
    frames = None
    correlation, max_corr = None, None
    step_size = None
    roll_cnt = None

    def __init__(self, signal1, signal2, normalized=True):
        self.signal1 = signal1
        self.signal2 = signal2
        self.len1 = len(signal1)
        self.len2 = len(signal2)
        self.normalized = normalized
        self.correlation = np.zeros(len(signal1) + len(signal2) - 1)
        self.sig1 = np.hstack((np.zeros(len(signal2) - 1), self.signal1, np.zeros(len(signal2) - 1)))
        self.sig2 = np.hstack((self.signal2, np.zeros(len(self.signal1) + len(self.signal2) - 2)))
        self.fig, self.axes = plt.subplots(3, 1, figsize=(12, 6))

    def start(self, save_gif=False, f_name='correlation'):
        '''start animation or save as gif'''
        self._init_animation()
        anim = FuncAnimation(self.fig, self._animate, frames=self.frames, blit=True, init_func=self._init_canvas)
        if save_gif:
            if not os.path.exists('vis'):
                os.makedirs('vis')
            anim.save(os.path.join('vis', f_name + '.gif'), codec='png', writer='imagemagick')
        else:
            plt.show()

    def _init_animation(self):
        ''' initialize animation'''
        self.frames = min(50, len(self.correlation))
        self.step_size = np.ceil(len(self.correlation) / self.frames).astype(np.int)
        self.roll_cnt = 0
        self.lines = [matplotlib.axes.Axes.plot] * 3
        # plot 1: show signal 1
        self.lines[0], = self.axes[0].plot([], [], linewidth=0.5)
        self.axes[0].set_xlim([0, len(self.sig1)])
        y_range = np.max(np.abs(self.signal1)) * 1.1
        self.axes[0].set_ylim([-y_range, y_range])
        if language == 'en':
            self.axes[0].set_title('Signal 1')
        else:
            self.axes[0].set_title('信号一')
        self.axes[0].axis('off')
        # plot 2: show signal 2
        self.lines[1], = self.axes[1].plot([], [], linewidth=0.5)
        self.axes[1].set_xlim([0, len(self.sig2)])
        y_range = np.max(np.abs(self.signal2)) * 1.1
        self.axes[1].set_ylim([-y_range, y_range])
        if language == 'en':
            self.axes[1].set_title('Signal 2')
        else:
            self.axes[1].set_title('信号二')
        self.axes[1].axis('off')
        # plot 3: show correlation
        self.lines[2], = self.axes[2].plot([], [], linewidth=0.5)
        self.axes[2].set_xlim([0, len(self.correlation) * 1.01])
        xcorr = circular_correlation(self.signal1, self.signal2)
        self.max_corr = max(np.abs(xcorr))
        self.axes[2].set_ylim([-1.1, 1.1])
        if language == 'en':
            self.axes[2].set_title('Correlation')
        else:
            self.axes[2].set_title('相关结果')
        self.axes[2].axis('off')
        self.point, = self.axes[2].plot([], [], 'ro', markersize=3)

    def _init_canvas(self):
        '''do nothing, return artists to be re-drawn'''
        return self.lines + [self.point]

    def _animate(self, i):
        '''perform animation step'''
        for j in range(self.step_size):
            self.correlation = np.roll(self.correlation, -1)        # shift history
            # normalized correlation
            len_corr = self._calculate_overlap()
            self.correlation[-1] = np.sum(self.sig1 * self.sig2) / len_corr / self.max_corr
            self.sig2 = np.roll(self.sig2, 1)
            self.roll_cnt = (self.roll_cnt + 1) % len(self.correlation)
        # update artists
        self.lines[0].set_data(range(len(self.sig1)), self.sig1)
        self.lines[1].set_data(range(len(self.sig2)), self.sig2)
        if self.correlation[-1] >= 0.95:
            self.lines[1].set_color('red')      # highlight when reaches peak
        else:
            self.lines[1].set_color('#1f77b4')
        self.lines[2].set_data(range(len(self.correlation)), self.correlation)
        self.point.set_data(len(self.correlation), self.correlation[-1])
        return self.lines + [self.point]

    def _calculate_overlap(self):
        if self.normalized:
            if self.roll_cnt < self.len1:
                len_corr = min(self.roll_cnt + 1, self.len2)
            else:
                len_corr = min(self.len1, self.len1 + self.len2 - self.roll_cnt - 1)
            return len_corr
        else:
            return min(self.len1, self.len2)


if __name__ == '__main__':

    save_vis = True

    # auto-correlation for random noise
    signal = np.random.randn(400)
    corr_anim = correlationAnimation(signal, signal, normalized=False)
    corr_anim.start(save_gif=save_vis, f_name='corr_noise_' + language)

    # auto-correlation for sine wave
    t = np.arange(0, 2e-3, 1 / 100e3)
    f1 = 1e3
    signal1 = np.sin(2 * np.pi * f1 * t)
    corr_anim = correlationAnimation(signal1, signal1)
    corr_anim.start(save_gif=save_vis, f_name='corr_sine_' + language)

    # circular correlation for sine wave
    signal2 = np.sin(2 * np.pi * (f1 + 200) * t)
    signal3 = np.sin(2 * np.pi * (f1 + 400) * t)
    xcorr1 = circular_correlation(signal1, signal1)
    xcorr2 = circular_correlation(signal1, signal2)
    xcorr3 = circular_correlation(signal1, signal3)
    plt.figure(figsize=(10, 6))
    # plt.rc('text', usetex=True)
    plt.plot(xcorr1, label=r'$\Delta$f=0Hz')
    plt.plot(xcorr2, label=r'$\Delta$f=200Hz')
    plt.plot(xcorr3, label=r'$\Delta$f=400Hz')
    plt.legend(loc='upper right', fontsize=16)
    plt.axis('off')
    if language == 'en':
        plt.title('Correlation of sine wave with different frequencies', fontsize=22)
    else:
        plt.title('不同频率的正弦波的相关性', fontsize=22)
    if not os.path.exists('vis'):
        os.makedirs('vis')
    if save_vis:
        plt.savefig(os.path.join('vis', 'corr_sine_' + language + '.png'))
    else:
        plt.show()

