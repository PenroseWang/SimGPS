"""
Tracking loop
"""

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from utils import sys_, func
from tracking.carrier_loop import carrierLoop, PLL
from tracking.code_loop import codeLoop
from signal.gps_signal import gpsSignal
from signal.ca_code import caCode
matplotlib.use('TkAgg')
language = 'cn'  # en: English or cn: Chinese
if language == 'cn':
    matplotlib.rcParams['font.family'] = ['Heiti TC']


class trackingLoop:
    '''
    Tracking loop, contains carrierLoop and codeLoop
    '''
    # system setting
    fs = None           # sampling frequency
    T_coh = None        # coherent time
    # filters
    carrier_loop, code_loop = None, None

    def __init__(self, T_coh, sv=1, fs=sys_.fs, fi_cap=sys_.fi, chip_cap=0):
        self.T_coh = T_coh
        self.fs = fs
        self.carrier_loop = carrierLoop(self.T_coh, fs=self.fs, fi_cap=fi_cap)
        self.code_loop = codeLoop(self.T_coh, sv, fs=self.fs, chip_cap=chip_cap)
        self.reset()

    def reset(self):
        self.code_loop.reset()
        self.carrier_loop.reset()

    def track(self, signal, duration, t0):
        # carrier NCO signal
        t, sig_cos, sig_sin = self.carrier_loop.generate_carrier_signals(duration, t0)
        # code NCO signal
        _, early, present, late = self.code_loop.generate_ca_signals(duration, t0,
                                                                     self.carrier_loop.fi_fll - sys_.fi)

        # coherent integration
        sig_i = signal * sig_cos
        sig_q = signal * sig_sin
        sig_i_e = sig_i * early
        sig_q_e = sig_q * early
        sig_i_p = sig_i * present
        sig_q_p = sig_q * present
        sig_i_l = sig_i * late
        sig_q_l = sig_q * late
        i, q = sum(sig_i_p), sum(sig_q_p)
        i_e, q_e = sum(sig_i_e), sum(sig_q_e)
        i_l, q_l = sum(sig_i_l), sum(sig_q_l)

        # update carrier loop
        error = self.carrier_loop.discriminator(i, q)
        self.carrier_loop.filter(error)
        self.carrier_loop.update_filter_status()

        # update code loop
        error = self.code_loop.discriminator(i_e, q_e, i_l, q_l)
        self.code_loop.filter(error)

        return t, sig_i_p, sig_q_p


class trackingLoopAnimation:
    '''
    Animation for tracking loop
    '''

    # signal settings
    fs, fi = None, None
    T_coh, duration = None, None
    sv, len_sv = None, None
    snr = None
    gps_signal_generator = None
    tracking_loops = None
    # animation attributes
    fig, axes, lines, points, texts = None, None, None, None, None
    frames = None
    i_first_30_chips = None
    len_history, histories = None, None

    def __init__(self, T_coh, duration, sv=None, fs=sys_.fs, fi=sys_.fi, snr=None):
        self.T_coh = T_coh
        self.duration = duration
        self.fs = fs
        self.fi = fi
        if sv is None:
            sv = [1]
        self.sv = sv
        self.len_sv = len(self.sv)
        self.tracking_loops = [trackingLoop] * self.len_sv
        self.snr = snr
        self.fig, self.axes = plt.subplots(self.len_sv + 1, 1, figsize=(8, 2 * (self.len_sv + 1)))

    def init_gps_signal_generator(self, t0=None, carrier_phase=None,
                                  doppler=None, doppler_drift=None):
        self.gps_signal_generator = gpsSignal(self.sv, fs=self.fs, fi=self.fi, t0=t0, carrier_phase=carrier_phase,
                                              doppler=doppler, doppler_drift=doppler_drift)

    def init_tracking_loop(self, fi_cap=None, chip_cap=None):
        if fi_cap is None:
            fi_cap = [sys_.fi] * self.len_sv
        else:
            if len(fi_cap) != self.len_sv:
                raise ValueError('Dimension of \'fi_cap\' and \'sv\' do not match.')
        if chip_cap is None:
            chip_cap = [0] * self.len_sv
        else:
            if len(chip_cap) != self.len_sv:
                raise ValueError('Dimension of \'chip_cap\' and \'sv\' do not match.')
        for i in range(self.len_sv):
            self.tracking_loops[i] = trackingLoop(self.T_coh, self.sv[i], fs=self.fs, fi_cap=fi_cap[i],
                                                  chip_cap=chip_cap[i])

    def start(self, save_gif=False, f_name='code_loop'):
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
        self.frames = int(self.duration // (3 * self.T_coh))
        self.len_history = 200
        self.lines = [matplotlib.axes.Axes.plot] * (self.len_sv * 2 + 1)
        self.texts = [matplotlib.axes.Axes.text] * self.len_sv
        self.histories = np.zeros((self.len_sv * 2, self.len_history))

        # plot 1: show GPS signal
        self.lines[0], = self.axes[0].plot([], [], linewidth=0.5)
        self.axes[0].set_xlim(0, 30 / sys_.f_ca)
        self.axes[0].set_ylim([-1.1 * self.len_sv, 1.1 * self.len_sv])
        if language == 'en':
            self.axes[0].set_title('GPS signal (with noise)')
        else:
            self.axes[0].set_title('GPS信号（含噪声）')
        self.axes[0].axis('off')
        self.i_first_30_chips = round(30 * (self.fs / sys_.f_ca))

        # remaining plots: show tracking results
        for i in range(self.len_sv):
            self.lines[2 * i + 1], = self.axes[i + 1].plot([], [], 'r', linewidth=0.5, label='I')
            self.lines[2 * i + 2], = self.axes[i + 1].plot([], [], 'b', linewidth=0.5, label='Q')
            if language == 'en':
                self.axes[i + 1].set_title('Code & carrier stripping for PRN' + str(self.sv[i]))
            else:
                self.axes[i + 1].set_title('PRN' + str(self.sv[i]) + '数据解调')
            self.axes[i + 1].set_xlim([-1, self.len_history + 1])
            self.axes[i + 1].axis('off')
            self.axes[i + 1].legend(loc='upper left')
            self.texts[i] = self.axes[i + 1].text(0.1, 0.1, '', transform=self.axes[i + 1].transAxes)

    def _init_canvas(self):
        '''do nothing, return artists to be re-drawn'''
        return self.lines + self.texts

    def _animate(self, i):
        '''perform animation step'''

        # run 3 iterations to reduce total frame number to 1/3
        for j in range(3):
            t, signal = self.gps_signal_generator.generate(self.T_coh, t0=(3 * i + j) * self.T_coh)
            # add noise
            if self.snr is not None:
                signal_n = func.awgn(signal, self.snr)
                signal_n /= max(signal_n)       # normalize to 1
            else:
                signal_n = signal

            # tracking for each satellite
            for k in range(self.len_sv):
                _, sig_i, sig_q = self.tracking_loops[k].track(signal_n, self.T_coh, t0=(3 * i + j) * self.T_coh)
                I, Q = np.mean(sig_i), np.mean(sig_q)
                self.histories[2 * k:2 * k + 2, 0:-1] = self.histories[2 * k:2 * k + 2, 1:]
                self.histories[2 * k, -1], self.histories[2 * k + 1, -1] = I, Q

        # plot 1: received signal and nco signal
        self.lines[0].set_data(t[:self.i_first_30_chips] - t[0], signal_n[:self.i_first_30_chips])
        # plot 2: tracking results for each sv
        for j in range(self.len_sv):
            self.lines[2 * j + 1].set_data(range(0, self.len_history), self.histories[2 * j])
            self.lines[2 * j + 2].set_data(range(0, self.len_history), self.histories[2 * j + 1])
            y_range = np.max(np.abs(self.histories[2 * j:2 * j + 2])) * 1.1
            self.axes[j + 1].set_ylim([-y_range, y_range])
            if self.tracking_loops[j].carrier_loop.filter_status == PLL:
                if language == 'en':
                    self.texts[j].set_text('status: PLL')
                else:
                    self.texts[j].set_text('状态: PLL')
            else:
                if language == 'en':
                    self.texts[j].set_text('status: FLL')
                else:
                    self.texts[j].set_text('状态: FLL')
        return self.lines + self.texts


if __name__ == '__main__':

    code_loop_anim = trackingLoopAnimation(T_coh=1e-3, duration=0.3, sv=[1, 2], snr=-5)
    code_loop_anim.init_gps_signal_generator(t0=[10000.5 / sys_.f_ca, 0], doppler=[125, 555])
    code_loop_anim.init_tracking_loop(fi_cap=[sys_.fi, sys_.fi + 500], chip_cap=[10000, 0])
    code_loop_anim.start(save_gif=True, f_name='tracking_loop_ani_' + language)

