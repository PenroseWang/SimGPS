"""
Carrier loop
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
matplotlib.use('TkAgg')
language = 'cn'  # en: English or cn: Chinese
if language == 'cn':
    matplotlib.rcParams['font.family'] = ['Heiti TC']

# define constants
PLL = 0
FLL = 1


class carrierLoop:
    '''
    carrierLoop class, contains PLL, FLL, and NCO
    '''

    # system setting
    fs = None       # sampling frequency
    fi_cap = None   # captured intermediate frequency
    mode = None     # 0: FLL/PLL, 1: PLL only, 2: FLL only
    T_coh = None    # coherent time
    # filters
    filter_pll, filter_fll = None, None
    filter_status = None        # PLL or FLL
    fi_fll = None               # last frequency tracked by FLL
    cnt_fll2pll = None          # count for switching FLL to PLL
    i_pre, q_pre = None, None   # used in fll discriminator
    filter_output_history = None
    # nco
    nco = None
    nco_step = None             # nco step size, determined by tracking frequency
    nco_phase, lut_index = None, None   # nco phase and lut index

    def __init__(self, T_coh, mode=0, fs=sys_.fs, fi_cap=sys_.fi, pll_bw=0.3, pll_order=3, fll_bw=0.3, fll_order=2):
        self.T_coh = T_coh
        self.mode = mode
        self.fs = fs
        self.fi_cap = fi_cap
        self.filter_pll = loopFilter(pll_bw, pll_order, self.T_coh)
        self.filter_fll = loopFilter(fll_bw, fll_order, self.T_coh)
        self.nco = NCO(fs=self.fs)
        self.reset()

    def reset(self, mode=None):
        if mode is not None:
            self.mode = mode
        self.filter_fll.reset_state()
        self.filter_pll.reset_state()
        self.filter_output_history = np.zeros(20)   # initialize as 0s
        # reset FLL discriminator history
        self.fi_fll = self.fi_cap
        self.i_pre, self.q_pre = None, None
        # reset nco
        self.nco_step = 0
        self.nco_phase, self.lut_index = 0, 0
        # reset flags
        if self.mode == 0 or self.mode == 2:
            # 0: FLL/PLL or 2: FLL only
            self.filter_status = FLL
            self.cnt_fll2pll = 0
        elif self.mode == 1:
            # 1: PLL only
            self.filter_status = PLL
        else:
            raise ValueError('Mode ' + str(self.mode) + ' is not supported.')

    def track(self, signal, duration, t0):
        t, sig_cos, sig_sin = self.generate_carrier_signals(duration, t0)
        # coherent integration
        i = np.sum(signal * sig_cos)
        q = np.sum(signal * sig_sin)
        error = self.discriminator(i, q)
        filter_output = self.filter(error)
        self.update_filter_status()
        return t, sig_cos, sig_sin, filter_output

    def generate_carrier_signals(self, duration, t0):
        t = np.arange(0, duration, 1 / self.fs) + t0
        if self.filter_status == PLL:
            self.nco_phase -= self.filter_output_history[-1] / (2 * np.pi) * self.nco.length * self.nco.len_lut
        else:
            # update frequency
            self.fi_fll -= self.filter_output_history[-1] / (2 * np.pi)

        nco_step = self.fi_fll / self.nco.resolution * self.nco.len_lut
        indices, self.nco_phase, self.lut_index = \
            self.nco.generate(nco_step, len(t), self.nco_phase, self.lut_index, self.nco.len_lut)
        cos_sig = self.nco.lut_cos[indices]
        sin_sig = self.nco.lut_sin[indices]
        return t, cos_sig, sin_sig

    def discriminator(self, i, q):
        if self.filter_status == PLL:
            # phase lock loop discriminator
            output = np.arctan(q/i)
        else:
            # frequency lock loop discriminator
            if self.i_pre is not None:
                p_cross = self.i_pre * q - self.q_pre * i
                p_dot = self.i_pre * i + self.q_pre * q
                norm_pre = np.sqrt(self.i_pre ** 2 + self.q_pre ** 2)
                norm = np.sqrt(i ** 2 + q ** 2)
                output = p_cross * np.sign(p_dot) / self.T_coh / norm / norm_pre
            else:
                output = 0
            self.i_pre, self.q_pre = i, q
        return output

    def filter(self, filter_input):
        if self.filter_status == PLL:
            filter_output = self.filter_pll.filter(filter_input)
        else:
            filter_output = self.filter_fll.filter(filter_input)
        self.filter_output_history[:-1] = self.filter_output_history[1:]
        self.filter_output_history[-1] = filter_output
        return filter_output

    def update_filter_status(self):
        if self.mode != 0:
            # no update for PLL-only and FLL-only mode
            return
        if self.filter_status == FLL:
            # accumulate counter if output is below threshold
            if np.abs(np.mean(self.filter_output_history)) < 5:
                self.cnt_fll2pll += 1
            else:
                self.cnt_fll2pll = 0
            # change to PLL after stabilized
            if self.cnt_fll2pll > 25:
                self.filter_status = PLL
                self.filter_output_history = np.zeros(20)   # reset to 0s
                self.filter_fll.reset_state()
        elif self.filter_status == PLL:
            if np.abs(self.filter_output_history[-1]) > np.pi:
                # loss of lock detected
                self.filter_status = FLL
                self.reset()


class carrierLoopAnimation:
    '''
    Animation for carrier loop
    '''

    # signal settings
    fs, fi = None, None
    T_coh, duration = None, None
    snr = None
    gps_signal_generator = None
    carrier_loop = None
    filter_history = 50
    filter_e = np.zeros(filter_history)     # record loop filter output
    # animation attributes
    fig, axes, lines, point, texts = None, None, None, None, None
    frames = None
    i_first_1_chip = None

    def __init__(self, T_coh, duration, fs=sys_.fs, snr=None):
        self.T_coh = T_coh
        self.duration = duration
        self.fs = fs
        self.snr = snr
        self.fig, self.axes = plt.subplots(2, 1, figsize=(8, 6))

    def init_gps_signal_generator(self, sv=None, fi=sys_.fi, doppler=None, doppler_drift=None):
        if sv is None:
            sv = [1]
        self.gps_signal_generator = gpsSignal(sv, fs=self.fs, fi=fi, carrier_phase=[np.pi / 3],
                                              doppler=doppler, doppler_drift=doppler_drift)

    def init_carrier_loop(self, fi_cap, mode):
        self.carrier_loop = carrierLoop(self.T_coh, fs=self.fs, mode=mode, fi_cap=fi_cap)

    def start(self, save_gif=False, f_name='carrier_loop'):
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
        self.lines = [matplotlib.axes.Axes.plot] * 3
        self.texts = [matplotlib.axes.Axes.text] * 3

        if language == 'en':
            self.lines[0], = self.axes[0].plot([], [], linewidth=0.5, label='recv')
            self.lines[1], = self.axes[0].plot([], [], linewidth=0.5, label='copy')
        else:
            self.lines[0], = self.axes[0].plot([], [], linewidth=0.5, label='接收')
            self.lines[1], = self.axes[0].plot([], [], linewidth=0.5, label='复制')
        self.axes[0].set_xlim(0, 11 / sys_.f_ca)
        self.axes[0].set_ylim([-1.1, 1.1])
        if language == 'en':
            self.axes[0].set_title('Carrier wave signal')
        else:
            self.axes[0].set_title('载波信号')
        self.i_first_10_chips = round(10 * (self.fs / sys_.f_ca))
        self.axes[0].axis('off')
        self.axes[0].legend(loc='upper right')

        self.lines[2], = self.axes[1].plot([], [], linewidth=0.5)
        self.point, = self.axes[1].plot([], [], 'ro', markersize=3)
        if language == 'en':
            self.axes[1].set_title('Loop filter output')
        else:
            self.axes[1].set_title('环路滤波器输出')
        self.axes[1].axis('off')
        self.axes[1].set_xlim([-1, self.filter_history])

        self.texts[0] = self.axes[1].text(0.02, 0.25, '', transform=self.axes[1].transAxes)
        self.texts[1] = self.axes[1].text(0.02, 0.15, '', transform=self.axes[1].transAxes)
        self.texts[2] = self.axes[1].text(0.02, 0.05, '', transform=self.axes[1].transAxes)

        self.i_first_chip = round(self.fs / sys_.f_ca)

    def _init_canvas(self):
        '''do nothing, return artists to be re-drawn'''
        return self.lines + [self.point] + self.texts

    def _animate(self, i):
        '''perform animation step'''
        # plot 1: received signal and nco signal
        t, signal = self.gps_signal_generator.generate(self.T_coh, t0=i * self.T_coh, has_ca=False)
        # add noise
        if self.snr is not None:
            signal_n = func.awgn(signal, self.snr)
            signal_n /= max(signal_n)       # normalize to 1
        else:
            signal_n = signal
        # tracking
        _, sig_i, _, filter_e = self.carrier_loop.track(signal_n, self.T_coh, t0=i * self.T_coh)
        self.lines[0].set_data(t[:10 * self.i_first_chip] - t[0], signal_n[:10 * self.i_first_chip])
        self.lines[1].set_data(t[:10 * self.i_first_chip] - t[0], sig_i[:10 * self.i_first_chip])

        # plot 2: filter output
        self.filter_e[0:-1] = self.filter_e[1:]
        self.filter_e[-1] = filter_e
        self.lines[2].set_data(range(0, self.filter_history), self.filter_e)
        self.point.set_data(self.filter_history - 1, self.filter_e[-1])
        y_max = max(abs(self.filter_e)) * 1.1
        self.axes[1].set_ylim([-y_max - 1e-10, y_max + 1e-10])
        # plot 2: loop filter status
        if self.carrier_loop.filter_status == PLL:
            if language == 'en':
                self.texts[0].set_text('status: PLL')
            else:
                self.texts[0].set_text('状态: PLL')
        else:
            if language == 'en':
                self.texts[0].set_text('status: FLL')
            else:
                self.texts[0].set_text('状态: FLL')

        # phase difference ground truth
        i_one_period = round(self.fs / self.carrier_loop.fi_fll)
        phi_diff = func.phase_diff_insensitive(signal[:i_one_period], sig_i[:i_one_period])
        if language == 'en':
            self.texts[1].set_text('phase diff: %.1f\u00b0' % phi_diff)
        else:
            self.texts[1].set_text('相位差异: %.1f\u00b0' % phi_diff)

        # frequency difference ground truth
        fi_recv = self.gps_signal_generator.fi + self.gps_signal_generator.doppler[0] + \
            self.gps_signal_generator.doppler_drift[0] * i * self.T_coh
        fi_copy = self.carrier_loop.fi_fll
        if language == 'en':
            self.texts[2].set_text('freq diff: %.1fHz' % (fi_recv - fi_copy))
        else:
            self.texts[2].set_text('频率差异: %.1fHz' % (fi_recv - fi_copy))

        return self.lines + [self.point] + self.texts


if __name__ == '__main__':

    # settings
    fi = sys_.f_ca
    fs = 20 * fi
    T_coh = 1e-3
    save_gif = True

    # # animation 1:
    # carrier_loop_anim = carrierLoopAnimation(T_coh, duration=0.1, fs=fs)
    # carrier_loop_anim.init_gps_signal_generator(fi=fi)
    # carrier_loop_anim.init_carrier_loop(fi, mode=1)     # PLL
    # carrier_loop_anim.start(save_gif, f_name='carrier_loop_ani1_' + language)

    # # animation 2:
    # carrier_loop_anim = carrierLoopAnimation(T_coh, duration=0.1, fs=fs)
    # carrier_loop_anim.init_gps_signal_generator(fi=fi, doppler=[100])
    # carrier_loop_anim.init_carrier_loop(fi, mode=1)     # PLL
    # carrier_loop_anim.start(save_gif, f_name='carrier_loop_ani2_' + language)

    # # animation 3:
    # carrier_loop_anim = carrierLoopAnimation(T_coh, duration=0.1, fs=fs)
    # carrier_loop_anim.init_gps_signal_generator(fi=fi, doppler=[100])
    # carrier_loop_anim.init_carrier_loop(fi, mode=2)     # FLL
    # carrier_loop_anim.start(save_gif, f_name='carrier_loop_ani3_' + language)

    # animation 4:
    carrier_loop_anim = carrierLoopAnimation(T_coh, duration=0.12, fs=fs)
    carrier_loop_anim.init_gps_signal_generator(fi=fi, doppler=[100])
    carrier_loop_anim.init_carrier_loop(fi, mode=0)     # FLL-PLL
    carrier_loop_anim.start(save_gif, f_name='carrier_loop_ani4_' + language)
