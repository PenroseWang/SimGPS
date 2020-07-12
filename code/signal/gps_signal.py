"""
Generate legacy GPS signal, with carrier, C/A, and data signal
"""

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from signal.ca_code import caCode
from utils import sys_
from utils.func import circular_correlation
matplotlib.use('TkAgg')


class gpsSignal:
    '''
    GPS signal class
    '''

    fs = None       # sampling frequency
    fi = None       # carrier wave intermediate frequency (IF)
    sv, len_sv = None, None     # visible satellites
    t0 = None       # initial received time for each satellite
    carrier_phase = None    # initial carrier phase offset for each satellite
    doppler = None          # doppler shift for each satellite
    doppler_drift = None    # doppler drift for each satellite
    data = None     # navigation data broadcast on each satellite
    ca_codes = None         # C/A code for each satellite

    def __init__(self, sv, fs=sys_.fs, fi=sys_.fi, t0=None, carrier_phase=None,
                 doppler=None, doppler_drift=None, data=None):
        '''
        :param sv:      list of visible satellites
        :param fs:      in Hz, sampling frequency
        :param fi:      in Hz, IF, use system setting as default
        :param carrier_phase:   in rad/s, initial carrier phase
        :param code_phase:      in chips, initial code phase
        :param doppler:         in Hz, doppler shift, default 0s
        :param doppler_drift:   in Hz/s, doppler drift, default 0s
        :param data:            list of lists, navigation data, default sys_.data
        '''
        self.sv = sv
        self.len_sv = len(sv)
        self.fs = fs
        self.fi = fi
        # initial code phase offset
        if t0 is None:
            self.t0 = np.zeros(self.len_sv)
        else:
            if len(t0) != self.len_sv:
                raise ValueError('Dimension of \'doppler_drift\' and \'sv\' do not match.')
            self.t0 = t0
        # doppler shift
        if doppler is None:
            self.doppler = np.zeros(self.len_sv)
        else:
            if len(doppler) != self.len_sv:
                raise ValueError('Dimension of \'doppler\' and \'sv\' do not match.')
            self.doppler = doppler
        # initial carrier phase offset
        if carrier_phase is None:
            self.carrier_phase = np.zeros(self.len_sv)
        else:
            if len(carrier_phase) != self.len_sv:
                raise ValueError('Dimension of \'carrier_phase\' and \'sv\' do not match.')
            self.carrier_phase = carrier_phase
        # doppler drift
        if doppler_drift is None:
            self.doppler_drift = np.zeros(self.len_sv)
        else:
            if len(doppler_drift) != self.len_sv:
                raise ValueError('Dimension of \'doppler_drift\' and \'sv\' do not match.')
            self.doppler_drift = doppler_drift
        # navigation data
        self.data = []
        if data is None:
            if self.len_sv > len(sys_.data):
                raise ValueError('Number of satellites exceeds sample navigation data length,'
                                 'please add more in sys_.py or pass your own argument.')
            for i in range(self.len_sv):
                self.data.append(sys_.data[i])
        else:
            if self.len_sv != len(data):
                raise ValueError('Dimension of \'data\' and \'sv\' do not match.')
            self.data = data
        # C/A code
        self.ca_codes = [caCode(x, self.fs) for x in self.sv]

    def generate(self, duration, t0=0, has_carrier=True, has_ca=True):
        '''
        Generate gps
        :param duration:    in sec
        :param t0:          in sec, start time
        :param has_carrier: modulate carrier wave if set True
        :param has_ca:      modulate C/A if set True
        :return:
            t:
            signal:
        '''
        t = np.arange(0, duration, 1 / self.fs) + t0
        signal = np.zeros(len(t))
        for i in range(self.len_sv):
            # ca signal
            ca_signal = self.ca_codes[i].generate_ca_signal(t + self.t0[i], self.doppler[i], self.doppler_drift[i])
            # carrier wave
            fi_signal = self.generate_carrier_signal(t + self.t0[i], self.carrier_phase[i],
                                                     self.doppler[i], self.doppler_drift[i])
            # data
            data_signal = self.generate_data_signal(t + self.t0[i], self.data[i], self.doppler[i],
                                                    self.doppler_drift[i])
            if has_carrier:
                data_signal = data_signal * fi_signal
            if has_ca:
                data_signal = data_signal * ca_signal
            signal += data_signal
        return t, signal

    @staticmethod
    def generate_data_signal(t, data=None, doppler=0, doppler_drift=0):
        if data is None:
            data = [0, 1]
        # phase =  f_ca + f_d + d_fd * t, where
        # f_d = k_data * doppler, dfd = k_data * doppler_drift
        phase = t * (sys_.f_data + sys_.k_data * (doppler + doppler_drift * t))
        t_code = np.mod(np.floor(phase), len(data)).astype(np.int)
        signal = 2 * np.array(data)[t_code] - 1   # convert from 0/1 to -1/1
        return signal

    def generate_carrier_signal(self, t, carrier_phase=0, doppler=0, doppler_drift=0):
        signal = np.cos(2 * np.pi * (self.fi + doppler + doppler_drift * t) * t + carrier_phase)
        return signal


class gpsSignalAnimation:
    '''
    Animation for plotting GPS signal characteristics
    '''

    # signal settings
    sv, len_sv, ca_codes = None, None, []
    duration, t0 = None, None
    gps_signal_generator = None
    # animation attributes
    fig, axes, lines, text = None, None, None, None
    frames = None

    def __init__(self, sv, duration, t0, doppler, doppler_drift):
        self.sv, self.len_sv = sv, len(sv)
        self.duration = duration
        self.t0 = t0
        self.gps_signal_generator = gpsSignal(sv, t0=t0, doppler=doppler, doppler_drift=doppler_drift)
        for i in range(self.len_sv):
            self.ca_codes.append(caCode(self.sv[i]))
        self.fig, self.axes = plt.subplots(self.len_sv + 1, 1, figsize=(12, 2 * (self.len_sv + 1)))

    def start(self, save_gif=False, f_name='gps_signal'):
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
        self.frames = int(self.duration // sys_.T_ca)       # show one period of C/A code as a frame
        # initialize drawing objects
        self.lines = [matplotlib.axes.Axes.plot] * (self.len_sv + 1)
        # frame
        self.axes[0].set_xlim(0, 25 / sys_.f_ca)
        self.axes[0].set_ylim([-self.len_sv, self.len_sv])
        self.axes[0].set_title('GPS signal (display first 25 chips)')
        self.axes[0].axis('off')
        self.lines[0], = self.axes[0].plot([], [], linewidth=0.5)
        self.text = self.axes[0].text(0.87, 0.9, '', transform=self.axes[0].transAxes)
        for i in range(self.len_sv):
            self.axes[i + 1].set_xlim([0, sys_.T_ca])
            self.axes[i + 1].set_ylim([-0.1, 0.1])
            self.axes[i + 1].set_title('Correlation with PRN' + str(self.sv[i]))
            self.axes[i + 1].axis('off')
            self.lines[i + 1], = self.axes[i + 1].plot([], [], linewidth=0.5)

    def _init_canvas(self):
        '''do nothing, return artists to be re-drawn'''
        return self.lines + [self.text]

    def _animate(self, i):
        '''perform animation step'''
        self.text.set_text('frame: %3d/%3d' % (i, self.frames))
        t, signal = self.gps_signal_generator.generate(sys_.T_ca, t0=i * sys_.T_ca)
        self.lines[0].set_data(t - t[0], signal)
        for k in range(self.len_sv):
            ca_signal = self.ca_codes[k].generate_ca_signal(t)
            corr = circular_correlation(signal, ca_signal)
            self.lines[k + 1].set_data(t - t[0], np.fft.fftshift(corr))

        return self.lines + [self.text]


if __name__ == '__main__':

    # Setting
    sv = [1, 2, 3]
    t0 = [0, 250 / sys_.f_ca, 0]
    doppler = [100, 0, 0]
    doppler_drift = [0, 0, 1e4]
    duration = 0.1

    # Animation
    gps_corr_anim = gpsSignalAnimation(sv, duration, t0, doppler, doppler_drift)
    gps_corr_anim.start(save_gif=False)

