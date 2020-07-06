"""
Generate legacy GPS signal, with carrier, C/A, and data signal
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from signal.generate_ca_code import caCode
from utils import sys_
from utils.func import circular_correlation
matplotlib.use('TkAgg')


class gpsSignal:
    '''
    GPS signal class
    '''

    fs = None       # sampling frequency
    fi = None       # carrier wave intermediate frequency (IF)
    sv = None       # visible satellites
    code_phase = None       # initial code phase for each satellite
    doppler = None          # doppler shift for each satellite
    doppler_drift = None    # doppler drift for each satellite
    data = None     # navigation data broadcast on each satellite

    def __init__(self, sv, fs=sys_.fs, fi=sys_.fi, code_phase=None, doppler=None, doppler_drift=None, data=None):
        '''
        :param sv:      list of visible satellites
        :param fs:      in Hz, sampling frequency
        :param fi:      in Hz, IF, use system setting as default
        :param code_phase:      in chips, initial code phase
        :param doppler:         in Hz, doppler shift, default 0s
        :param doppler_drift:   in Hz/s, doppler drift, default 0s
        :param data:            list of lists, navigation data, default sys_.data
        '''
        self.sv = sv
        self.fs = fs
        self.fi = fi
        # doppler shift
        if doppler is None:
            self.doppler = np.zeros(len(self.sv))
        else:
            if len(doppler) != len(sv):
                raise ValueError('Dimension of \'doppler\' and \'sv\' do not match.')
            self.doppler = doppler
        # initial code phase
        if code_phase is None:
            self.code_phase = np.zeros(len(self.sv))
        else:
            if len(code_phase) != len(sv):
                raise ValueError('Dimension of \'doppler_drift\' and \'sv\' do not match.')
            self.code_phase = code_phase
        # doppler drift
        if doppler_drift is None:
            self.doppler_drift = np.zeros(len(self.sv))
        else:
            if len(doppler_drift) != len(sv):
                raise ValueError('Dimension of \'doppler_drift\' and \'sv\' do not match.')
            self.doppler_drift = doppler_drift
        # navigation data
        self.data = []
        if data is None:
            if len(sv) > len(sys_.data):
                raise ValueError('Number of satellites exceeds sample navigation data length,'
                                 'please add more in sys_.py or pass your own argument.')
            for i in range(len(sv)):
                self.data.append(sys_.data[i])
        else:
            if len(sv) != len(data):
                raise ValueError('Dimension of \'data\' and \'sv\' do not match.')
            self.data = data

    def generate(self, duration, t0=0):
        '''
        Generate gps
        :param duration:    in sec
        :param t0:          in sec, start time
        :return:
            t:
            signal:
        '''
        t = np.arange(t0, t0 + duration, 1 / self.fs)
        signal = np.zeros(len(t))
        for i in range(len(self.sv)):
            # ca signal
            _, ca_signal = caCode(self.sv[i]).generate_ca_signal(duration, t0, self.code_phase[i], self.doppler[i],
                                                                 self.doppler_drift[i], self.fs)
            # carrier wave
            fi_signal = np.sin(2 * np.pi * (self.fi + self.doppler[i] + self.doppler_drift[i] * t) * t)
            # data
            data_signal = self._generate_data_signal(self.data[i], duration, t0, self.doppler[i], self.doppler_drift[i])
            signal += ca_signal * fi_signal * data_signal
        return t, signal

    def _generate_data_signal(self, data, duration, t0=0, doppler=0, doppler_drift=0):
        t = np.arange(t0, t0 + duration, 1 / self.fs)
        # phase =  f_ca + f_d + d_fd * t, where
        # f_d = k_data * doppler, dfd = k_data * doppler_drift
        phase = t * (sys_.f_data + sys_.k_data * (doppler + doppler_drift * t))
        t_code = np.mod(np.floor(phase), len(data)).astype(np.int)
        signal = 2 * np.array(data)[t_code] - 1   # convert from 0/1 to -1/1
        return signal


class gpsSignalAnimation:
    """
    Animation for plotting GPS signal characteristics
    """

    # signal settings
    sv, len_sv = None, None
    duration = None
    T_coh = None
    gps_signal_generator = None
    ca_codes = []
    # plot attributes
    fig, axes, lines, text = None, None, None, None
    frames = None
    # settings
    save_gif = False

    def __init__(self, sv, duration, code_phase, doppler, doppler_drift, save_gif=False):
        self.sv, self.len_sv = sv, len(sv)
        self.duration = duration
        self.T_coh = 1e-3
        self.gps_signal_generator = gpsSignal(sv, code_phase=code_phase, doppler=doppler, doppler_drift=doppler_drift)
        for i in range(self.len_sv):
            self.ca_codes.append(caCode(self.sv[i]))
        self.save_gif = save_gif
        self.fig, self.axes = plt.subplots(self.len_sv + 1, 1, figsize=(12, 2 * (self.len_sv + 1)))
        self.lines = [matplotlib.axes.Axes.plot] * (self.len_sv + 1)
        self.frames = int(self.duration // self.T_coh)

    def start(self):
        '''start animation or save as gif'''

        anim = FuncAnimation(self.fig, self._animate, frames=self.frames, blit=True, init_func=self._init_animation)
        if self.save_gif:
            anim.save('gps_corr.gif', codec='png', writer='imagemagick')
        else:
            plt.show()

    def _init_animation(self):
        '''initialize animation'''

        self.axes[0].set_xlim(0, 25 / sys_.f_ca)
        self.axes[0].set_ylim([-self.len_sv, self.len_sv])
        self.axes[0].set_title('GPS signal (display first 25 chips)')
        self.axes[0].axis('off')
        self.lines[0], = self.axes[0].plot([], [], linewidth=0.5)
        self.text = self.axes[0].text(0.87, 0.9, '', transform=self.axes[0].transAxes)
        for i in range(self.len_sv):
            self.axes[i + 1].set_xlim([0, self.T_coh])
            self.axes[i + 1].set_ylim([-0.1, 0.1])
            self.axes[i + 1].set_title('Correlation with PRN' + str(self.sv[i]))
            self.axes[i + 1].axis('off')
            self.lines[i + 1], = self.axes[i + 1].plot([], [], linewidth=0.5)

        return self.lines + [self.text]

    def _animate(self, i):
        '''perform animation step'''

        self.text.set_text('frame: %3d/%3d' % (i, self.frames))
        t, signal = self.gps_signal_generator.generate(self.T_coh, t0=i * self.T_coh)
        self.lines[0].set_data(t - t[0], signal)
        for k in range(self.len_sv):
            _, ca_signal = self.ca_codes[k].generate_ca_signal(self.T_coh, t0=i * self.T_coh)
            corr = circular_correlation(signal, ca_signal)
            self.lines[k + 1].set_data(t - t[0], np.fft.fftshift(corr))

        return self.lines + [self.text]


if __name__ == '__main__':

    # Setting
    sv = [1, 2, 3]
    code_phase = [0, 250, 0]
    doppler = [0, 0, 300]
    doppler_drift = [0, 0, 5000]
    duration = 0.1
    save_gif = True

    # Animation
    gps_corr_anim = gpsSignalAnimation(sv, duration, code_phase, doppler, doppler_drift, save_gif)
    gps_corr_anim.start()

