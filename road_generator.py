import numpy as np
import random
import math
from scipy import signal

class RoadGenerator():
    def __init__(self):
        self.L  = 300       # Length of Road Profile (m)
        self.B  = 0.1       # Sampling Interval (m)
        self.dn = 1/self.L  # Frequency Band (1/m)
        self.n0 = 0.1       # Spatial Frequency (cycles/m)

    def createGeneralRoad(self, Vel_from, Vel_to, k1):
        # Vel_from : Initial Velocity (kph)
        # Vel_to : Final Velocity (kph)
        # k1: from ISO 8608, 3 is very rough road

        # L: Length of Road Profile (m)
        # B: Sampling Interval (m)
        # dn: Frequency Band (1/m)
        # n0: Spatial Frequency (cycles/m)

        L = self.L
        B = self.B
        dn= self.dn
        n0= self.n0

        # N: Number of data points
        N = L/B

        # Spatial Frequency Band (1/m)
        n = np.arange(dn, N*dn+dn, dn)

        # Abscissa Variable from 0 to L (x-Coordinate, m)
        x = np.arange(0, L-B+B, B)

        # Amplitude for Road  Class
        Amp1 = math.sqrt(dn) * math.pow(2, k1) * (1e-3) * (n0/n)

        def gen():
            # Random Phase Angle
            phi1 = 2 * math.pi * np.random.rand(n.shape[0])

            # Road hight (m)
            z = np.zeros(x.shape[0])
            for i in range(x.shape[0]):
                z[i] = np.sum(Amp1 * np.cos(2 * math.pi * n * x[i] + phi1))

            # Removing linear trend
            z = signal.detrend(z)

            # 처음과 마지막 구간 0으로 수렴하도록 윈도우 적용
            z = z * signal.tukey(z.shape[0], 0.1)

            return z

        z_LH = gen()
        z_RH = gen()

        # Random velocity
        v = np.full(z_LH.shape, (Vel_from + (Vel_to - Vel_from) * random.random()))

        # kph to mps
        v_mps = v / 3.6

        # End Time
        t_end = x / v_mps[0]

        return x, z_LH, z_RH, v, t_end

    def createBumpRoad(self, Vel_from, Vel_to, width=None, height=None):
        if width is None:
            width = 3000 + 600 * random.random()
        if height is None:
            height = 80  + 20 * random.random()
        
        B = self.B

        # milimeter to meter
        width = width/1000
        height= height/1000

        temp = np.arange(-width/2, width/2+B, B)

        # Road Height (m)
        z = 1 - (np.power(temp, 2) / (np.power(width/2, 2)))
        z = z.clip(min=0)
        z = np.sqrt(z) * height

        add_zeros_before = 10

        # Road Distance (m)
        x = np.arange(0, (z.shape[0] + add_zeros_before) * B, B)
        
        # Add zero range
        add_zero_after = 10
        z = np.concatenate([
            np.zeros((add_zero_after,)),
            z,
            np.zeros((add_zeros_before-add_zero_after,))
        ])

        # Random velocity
        v = np.full(z.shape, (Vel_from + (Vel_to - Vel_from) * random.random()))

        # kph to mps
        v_mps = v / 3.6

        # End Time
        t_end = x / v_mps[0]

        # Same roads for left/right tires
        return x, z, z, v, t_end


if __name__ == "__main__":
    rg = RoadGenerator()
    # rg.createGeneralRoad(30, 50, 1.5)
    rg.createBumpRoad(30, 40, 3000, 100)
