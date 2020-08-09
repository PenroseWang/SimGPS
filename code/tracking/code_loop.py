"""
Code loop
"""

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from utils import sys_, func
from tracking.nco import NCO
from tracking.loop_filter import loopFilter
from signal.gps_signal import gpsSignal
from signal.ca_code import caCode
matplotlib.use('TkAgg')
language = 'cn'  # en: English or cn: Chinese
if language == 'cn':
    matplotlib.rcParams['font.family'] = ['Heiti TC']


class codeLoop:
    '''
    codeLoop class, contains DLL and NCO
    '''

    # system setting
    fs = None           # sampling frequency
    fi_cap = None       # captured intermediate frequency
    T_coh = None        # coherent time
    # C/A code
    ca_code_table = None
    # filter
    filter_dll = None
    dll_bw = None
    filter_output_history = None
    # nco
    nco = None
    nco_step = None             # nco step size, determined by tracking frequency
    nco_phase, lut_index = None, None   # nco phase and lut index

    def __init__(self, T_coh, sv=None, fs=sys_.fs, chip_cap=0, dll_bw=0.1, dll_order=2):
        self.T_coh = T_coh
        if sv is None:
            sv = 1
        self.ca_code_table = caCode(sv).code
        self.fs = fs
        self.chip_cap = chip_cap
        self.filter_dll = loopFilter(dll_bw, dll_order, self.T_coh)
        self.nco = NCO(fs=self.fs)
        self.reset()

    def reset(self):
        self.filter_dll.reset_state()
        self.filter_output_history = np.zeros(20)   # initialize as 0s
        # reset nco
        self.nco_step = 0
        self.nco_phase, self.lut_index = 0, self.chip_cap

    def track(self, signal_i, signal_q, duration, t0, doppler=0):
        t, early, present, late = self.generate_ca_signals(duration, t0, doppler)
        # coherent integration
        i_e = np.sum(signal_i * early)
        i_l = np.sum(signal_i * late)
        q_e = np.sum(signal_q * early)
        q_l = np.sum(signal_q * late)
        error = self.discriminator(i_e, q_e, i_l, q_l)
        filter_output = self.filter(error)
        return t, early, present, late, filter_output

    def generate_ca_signals(self, duration, t0, doppler=0):
        t = np.arange(0, duration, 1 / self.fs) + t0
        # add carrier aiding
        nco_step = (sys_.f_ca + doppler * sys_.k_ca) / self.nco.resolution
        # update LUT index
        self.lut_index += self.nco_phase / self.nco.length - self.filter_output_history[-1]
        # early
        nco_phase_e, lut_idx_e = self._get_nco_phase(self.lut_index - 0.5)
        indices_e, _, _ = self.nco.generate(nco_step, len(t), nco_phase_e,
                                            lut_idx_e, len(self.ca_code_table))
        # late
        nco_phase_l, lut_idx_l = self._get_nco_phase(self.lut_index + 0.5)
        indices_l, _, _ = self.nco.generate(nco_step, len(t), nco_phase_l,
                                            lut_idx_l, len(self.ca_code_table))
        # present
        nco_phase_p, lut_idx_p = self._get_nco_phase(self.lut_index)
        indices_p, self.nco_phase, self.lut_index = \
            self.nco.generate(nco_step, len(t), nco_phase_p, lut_idx_p, len(self.ca_code_table))
        sig_e = 2 * self.ca_code_table[indices_e] - 1   # convert 0/1 to -1/1
        sig_p = 2 * self.ca_code_table[indices_p] - 1
        sig_l = 2 * self.ca_code_table[indices_l] - 1
        return t, sig_e, sig_p, sig_l

    def _get_nco_phase(self, lut_index):
        lut_index, remainder = divmod(lut_index, 1)
        nco_phase = remainder * self.nco.length
        return nco_phase, int(lut_index)

    @staticmethod
    def discriminator(i_e, q_e, i_l, q_l):
        early = np.sqrt(i_e ** 2 + q_e ** 2)
        late = np.sqrt(i_l ** 2 + q_l ** 2)
        output = 0.5 * (early - late) / (early + late)
        return output

    def filter(self, filter_input):
        filter_output = self.filter_dll.filter(filter_input)
        self.filter_output_history[:-1] = self.filter_output_history[1:]
        self.filter_output_history[-1] = filter_output
        return filter_output


class codeLoopAnimation:
    '''
    Animation for carrier loop
    '''

    # signal settings
    fs, fi = None, None
    T_coh, duration = None, None
    snr = None
    gps_signal_generator = None
    code_loop = None
    filter_history = 50
    filter_e = np.zeros(filter_history)     # record loop filter output
    # animation attributes
    fig, axes, lines, points, texts = None, None, None, None, None
    frames = None
    i_first_1_chip = None

    def __init__(self, T_coh, duration, fs=sys_.fs, fi=sys_.fi, snr=None):
        self.T_coh = T_coh
        self.duration = duration
        self.fs = fs
        self.fi = fi
        self.snr = snr
        self.fig = plt.figure(figsize=(8, 6))
        self.axes = [matplotlib.axes.Axes] * 3
        self.axes[0] = self.fig.add_subplot(211)
        self.axes[1] = self.fig.add_subplot(223)
        self.axes[2] = self.fig.add_subplot(224)

    def init_gps_signal_generator(self, sv=None, t0=None, carrier_phase=None,
                                  doppler=None, doppler_drift=None):
        if sv is None:
            sv = [1]
        if t0 is None:
            t0 = [0.5 / sys_.f_ca]    # default setting, shift half a chip
        self.gps_signal_generator = gpsSignal(sv, fs=self.fs, fi=self.fi, t0=t0, carrier_phase=carrier_phase,
                                              doppler=doppler, doppler_drift=doppler_drift)

    def init_code_loop(self, chip_cap=0):
        self.code_loop = codeLoop(self.T_coh, fs=self.fs, chip_cap=chip_cap)

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
        self.frames = int(self.duration // self.T_coh)
        self.lines = [matplotlib.axes.Axes.plot] * 4
        self.points = [matplotlib.axes.Axes.plot] * 4
        self.texts = [matplotlib.axes.Axes.text] * 3

        if language == 'en':
            self.lines[0], = self.axes[0].plot([], [], linewidth=0.5, label='recv')
            self.lines[1], = self.axes[0].plot([], [], linewidth=0.5, label='nco')
        else:
            self.lines[0], = self.axes[0].plot([], [], linewidth=0.5, label='接收')
            self.lines[1], = self.axes[0].plot([], [], linewidth=0.5, label='复制')

        self.axes[0].set_xlim(0, 28 / sys_.f_ca)
        self.axes[0].set_ylim([-1.1, 1.1])
        if language == 'en':
            self.axes[0].set_title('PRN Code (display first 25 chips)')
        else:
            self.axes[0].set_title('伪码信号(前25个码片)')

        self.axes[0].axis('off')
        self.axes[0].legend()

        self.lines[2], = self.axes[1].plot([], [], linewidth=0.5)
        self.points[0], = self.axes[1].plot([], [], 'ro', markersize=3)
        if language == 'en':
            self.axes[1].set_title('Loop filter output')
        else:
            self.axes[1].set_title('环路滤波器输出')

        self.axes[1].axis('off')
        self.axes[1].set_xlim([-1, self.filter_history])

        self.texts[0] = self.axes[2].text(0, -2, 'E')
        self.texts[1] = self.axes[2].text(0, -2, 'P')
        self.texts[2] = self.axes[2].text(0, -2, 'L')
        self.lines[3], = self.axes[2].plot([-1, 0, 1], [0, 1, 0], linewidth=0.5)
        self.points[1], = self.axes[2].plot([], [], 'ro', linewidth=0.5)
        self.points[2], = self.axes[2].plot([], [], 'ro', linewidth=0.5)
        self.points[3], = self.axes[2].plot([], [], 'ro', linewidth=0.5)
        if language == 'en':
            self.axes[2].set_title('Code correlation for early, present, and late signal')
        else:
            self.axes[2].set_title('超前(E), 即时(P), 滞后(L)信号相关结果')
        self.axes[2].axis('off')

        self.i_first_chip = round(self.fs / sys_.f_ca)

    def _init_canvas(self):
        '''do nothing, return artists to be re-drawn'''
        return self.lines + self.points + self.texts

    def _animate(self, i):
        '''perform animation step'''
        # plot 1: received signal and nco signal
        t, signal = self.gps_signal_generator.generate(self.T_coh, t0=i * self.T_coh, has_carrier=False)
        # add noise
        if self.snr is not None:
            signal_n = func.awgn(signal, self.snr)
            signal_n /= max(signal_n)       # normalize to 1
        else:
            signal_n = signal
        # tracking
        _, sig_e, sig_p, sig_l, filter_e = self.code_loop.track(signal_n, 0, self.T_coh, t0=i * self.T_coh)
        self.lines[0].set_data(t[:25 * self.i_first_chip] - t[0], signal_n[:25 * self.i_first_chip])
        self.lines[1].set_data(t[:25 * self.i_first_chip] - t[0], sig_p[:25 * self.i_first_chip])

        # plot 2: filter output
        self.filter_e[0:-1] = self.filter_e[1:]
        self.filter_e[-1] = filter_e
        self.lines[2].set_data(range(0, self.filter_history), self.filter_e)
        self.points[0].set_data(self.filter_history - 1, self.filter_e[-1])
        y_max = max(abs(self.filter_e)) * 1.1
        self.axes[1].set_ylim([-y_max - 1e-10, y_max + 1e-10])

        # plot 3: code correlation
        corr_e = np.abs(np.mean(signal * sig_e))
        corr_p = np.abs(np.mean(signal * sig_p))
        corr_l = np.abs(np.mean(signal * sig_l))
        if corr_e > corr_l:
            x_e, x_p, x_l = 1 - corr_e, 1 - corr_p, 1 - corr_l
            if corr_p > 0.5:
                x_e = -x_e
            else:
                x_l = -100      # out of frame
        else:
            x_e, x_p, x_l = corr_e - 1, corr_p - 1, corr_l - 1
            if corr_p > 0.5:
                x_l = -x_l
            else:
                x_e = 100       # out of frame

        self.points[1].set_data(x_e, corr_e)
        self.points[2].set_data(x_p, corr_p)
        self.points[3].set_data(x_l, corr_l)
        self.texts[0].set_position([x_e - 0.02, corr_e - 0.1])
        self.texts[1].set_position([x_p - 0.02, corr_p - 0.1])
        self.texts[2].set_position([x_l - 0.02, corr_l - 0.1])

        return self.lines + self.points + self.texts


if __name__ == '__main__':

    chip_delay = 10.9
    code_loop_anim = codeLoopAnimation(T_coh=1e-3, duration=0.1)
    code_loop_anim.init_gps_signal_generator(t0=[chip_delay / sys_.f_ca])
    code_loop_anim.init_code_loop(chip_cap=np.floor(chip_delay))
    code_loop_anim.start(save_gif=True, f_name='code_loop_ani_' + language)

