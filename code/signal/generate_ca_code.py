"""
Generate C/A code and analyze auto-correlation and cross-correlation characteristics
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from utils.func import circular_correlation
from utils import sys_
matplotlib.use('TkAgg')


class caCode:
    '''
    C/A code class
    '''
    prn = None
    code = None

    def __init__(self, sv):
        self.prn = sv
        self._generate_ca_code()

    def generate_ca_signal(self, duration, t0=0, phase0=0, doppler=0, doppler_drift=0, fs=sys_.fs):
        '''
        Generate legacy GPS C/A signal
        :param duration:        in sec
        :param t0:              in sec, start time
        :param doppler:         in Hz, carrier doppler shift frequency
        :param doppler_drift:   in Hz/s, carrier doppler drift frequency
        :param fs:              in Hz, sampling frequency
        :return:
            t:          sampling time, sampling frequency set as in utils.py
            ca_signal:  sampled ca signal
        '''
        t = np.arange(t0, t0 + duration, 1 / fs)
        # phase =  f_ca + f_d + d_fd * t
        # f_d = k_ca * doppler, where k_ca is carrier aiding factor
        # dfd = k_ca * doppler_drift
        phase = t * (sys_.f_ca + sys_.k_ca * (doppler + doppler_drift * t)) + phase0
        t_code = np.mod(np.floor(phase), len(self.code)).astype(np.int)
        ca_signal = 2 * self.code[t_code] - 1   # convert 0/1 to -1/1
        return t, ca_signal

    def _generate_ca_code(self):
        g1 = self.generate_m_code(10, [3, 10])
        g2 = self.generate_m_code(10, [2, 3, 6, 8, 9, 10])
        g2_shift = np.roll(g2, sys_.delay[self.prn - 1])        # C/A code delay
        self.code = np.logical_xor(g1, g2_shift).astype(np.int)

    @staticmethod
    def generate_m_code(num_bits, taps):
        '''
        Generate M code using linear feedback shift register (LFSR)
        :param num_bits:    n bits of memory
        :param taps:        bits that influences LFSR input, determines the feedback polynomials
        :return:
            m_code:         pseudorandom sequence
        '''

        N = 2 ** num_bits - 1
        register_states = np.ones(num_bits).astype(np.int)   # initialize as all 1s
        m_code = np.empty(N).astype(np.int)
        for i in range(N):
            m_code[i] = register_states[-1]
            taps_states = [register_states[x - 1] for x in taps]
            # shift 1 bit to right
            register_states[1:] = register_states[0:-1]
            register_states[0] = np.logical_xor.reduce(taps_states)

        return m_code


class caCodeAnimation:
    """
    Animation for plotting C/A code correlation characteristics
    """

    # signals
    t, sig1, sig2, corr = None, None, None, None
    # plot attributes
    fig, axes = None, None
    line0, line1, line2 = None, None, None
    point0, point1, point2 = None, None, None
    # settings
    save_gif = False
    sample_interval = 33 * round(sys_.fs / sys_.f_ca)       # move every 33 chips

    def __init__(self, t, signal1, signal2, save_gif=False):
        self.t = t
        self.sig1 = signal1
        self.sig2 = signal2
        self.corr = circular_correlation(self.sig1, self.sig2)
        self.save_gif = save_gif
        self.fig, self.axes = plt.subplots(3, 1, figsize=(10, 6))

    def start(self, title1, title2, title3):
        '''start animation or save as gif'''
        self.axes[0].axis('off')
        self.axes[1].axis('off')
        self.axes[2].axis('off')
        self.axes[0].set_title(title1)
        self.axes[1].set_title(title2)
        self.axes[2].set_title(title3)
        self.line0, = self.axes[0].plot([], [], linewidth=0.5)
        self.line1, = self.axes[1].plot([], [], linewidth=0.5)
        self.line2, = self.axes[2].plot([], [], linewidth=0.5)
        self.point0, = self.axes[0].plot([], [], 'bo', markersize=3)
        self.point1, = self.axes[1].plot([], [], 'bo', markersize=3)
        self.point2, = self.axes[2].plot([], [], 'ro', markersize=3)
        anim = FuncAnimation(self.fig, self._animate, frames=len(t) // self.sample_interval * 2,
                             blit=True, init_func=self._init_animation)
        if self.save_gif:
            anim.save(title3 + '.gif', codec='png', writer='imagemagick')
        else:
            plt.show()

    def _init_animation(self):
        '''initialize animation'''
        self.line0.set_data(self.t, self.sig1)
        self.point0.set_data(0, self.sig1[0])
        self.line1.set_data([], [])
        self.point1.set_data(0, self.sig2[0])
        self.line2.set_data([], [])
        self.point2.set_data([], [])

        margin = t[-1] * 0.1
        self.axes[0].set_xlim([-margin, t[-1] + margin])
        self.axes[0].set_ylim([-1.1, 1.1])
        self.axes[1].set_xlim([-margin, t[-1] + margin])
        self.axes[1].set_ylim([-1.1, 1.1])
        self.axes[2].set_xlim([-margin, t[-1] + margin])
        self.axes[2].set_ylim([-0.1, 1.1])
        return self.line2, self.line2, self.point1, self.point2

    def _animate(self, i):
        '''perform animation step'''
        shift = self.sample_interval * i % len(self.t)
        self.line1.set_data(self.t, np.roll(self.sig2, -shift))
        self.point1.set_data(self.t[-shift], self.sig2[0])
        if self.sample_interval * i >= len(self.t):
            self.line2.set_data(self.t, np.roll(self.corr, -shift))
        else:
            self.line2.set_data(self.t[-shift - 1:], np.roll(self.corr, -shift)[-shift - 1:])
        self.point2.set_data(self.t[-1], self.corr[shift - 1])
        return self.line1, self.line2, self.point1, self.point2


if __name__ == '__main__':

    # Parameters
    duration = sys_.T_ca    # one period of C/A signal
    sv1, sv2 = 1, 2

    # C/A code
    PRN1 = caCode(sv1)
    _, PRN1_signal = PRN1.generate_ca_signal(duration)
    PRN2 = caCode(sv2)
    t, PRN2_signal = PRN2.generate_ca_signal(duration)

    # Animation
    save_gif = False
    auto_corr = caCodeAnimation(t, PRN1_signal, PRN1_signal, save_gif)
    auto_corr.start('PRN' + str(sv1), 'PRN' + str(sv2), 'auto-correlation')

    cross_corr = caCodeAnimation(t, PRN1_signal, PRN2_signal, save_gif)
    cross_corr.start('PRN' + str(sv1), 'PRN' + str(sv2), 'cross-correlation')


