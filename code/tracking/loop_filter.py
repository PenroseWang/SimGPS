"""
Frequency lock loop (FLL)
"""

import numpy as np


class loopFilter:
    '''
    loopFilter class
    '''

    bandwidth = None    # filter bandwidth
    order = None        # filter order, typical value: 1， 2 or 3
    x, y = None, None       # filter states
    b_x, b_y = None, None   # filter coefficients

    def __init__(self, bandwidth, order, T_coh):
        self.bandwidth = bandwidth
        self.order = order
        self.init_filter(T_coh)

    def init_filter(self, T_coh):
        if self.order == 1:
            omega = self.bandwidth / 0.25
            self.b_x = np.array([omega])
            self.b_y = np.array([0])
        elif self.order == 2:
            omega = self.bandwidth / 0.53
            a2 = 1.414
            b0 = a2*omega + T_coh * omega ** 2 / 2
            b1 = -a2*omega + T_coh * omega ** 2 / 2
            self.b_x = np.array([b0, b1])
            self.b_y = np.array([0, 1])
        elif self.order == 3:
            omega = self.bandwidth / 0.7845
            a3 = 1.1
            b3 = 2.4
            k1 = T_coh ** 2 * omega ** 3 / 4
            k2 = a3 * T_coh * omega ** 2 / 2
            k3 = b3 * omega
            b0 = k1 + k2 + k3
            b1 = 2 * (k1 - k3)
            b2 = k1 - k2 + k3
            self.b_x = np.array([b0, b1, b2])
            self.b_y = np.array([0, 2, -1])
        else:
            raise ValueError('Filter order ' + str(self.order) + ' not supported for FLL.')
        self.reset_state()

    def reset_state(self):
        self.x = np.zeros(self.order)
        self.y = np.zeros(self.order)

    def filter(self, filter_in):

        self.x[1:] = self.x[0:-1]   # discard oldest input
        self.x[0] = filter_in       # consume input
        self.y[1:] = self.y[0:-1]   # discard oldest output
        self.y[0] = np.sum(self.b_y * self.y + self.b_x * self.x)   # filtering
        return self.y[0]

