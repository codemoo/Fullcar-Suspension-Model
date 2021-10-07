import numpy as np
from scipy.signal import cont2discrete
import casadi as ca

class FullCarSuspension:

    def __init__(self):
        #[z dz phi dphi theta dtheta z_LF dz_LF z_RF dz_RF z_LR dz_LR z_RR dz_RR]
        # 1 is front, 2 is rear
        Ks_RF = 26700
        Ks_LF = 26700
        Ks_RR = 23150
        Ks_LR = 23150
        Cs_RF = 50
        Cs_LF = 50
        Cs_RR = 50
        Cs_LR = 50
        L = 2650*0.001
        W = 1568*0.001

        self.L = L
        self.W = W

        l_CF = L/2
        l_CR= L/2
        l_CRR = W/2
        l_CLL = W/2
        Kw_RF = 203464.1
        Kw_LF = 203464.1
        Kw_RR = 203464.1
        Kw_LR = 203464.1
        ms = 1106.284
        Iyy = 1447.5
        Ixx = 438.7
        Izz = 1671.2
        mw_RF = 101.488/2
        mw_LF = 101.488/2
        mw_RR = 86.718/2
        mw_LR = 86.718/2

        C_min = 300

        self.n_states = 14
        self.n_control = 4
        self.Ts = 0.01

        A = np.array([
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [
                (-Ks_RF-Ks_LF-Ks_RR-Ks_LR)/ms,
                (-Cs_RF-Cs_LF-Cs_RR-Cs_LR)/ms,
                (Ks_RF*l_CF+Ks_LF*l_CF-Ks_RR*l_CR-Ks_LR*l_CR)/ms,
                (Cs_RF*l_CF+Cs_LF*l_CF-Cs_RR*l_CR-Cs_LR*l_CR)/ms,
                (Ks_RF*l_CRR-Ks_LF*l_CLL+Ks_RR*l_CRR-Ks_LR*l_CLL)/ms,
                (Cs_RF*l_CRR-Cs_LF*l_CLL+Cs_RR*l_CRR-Cs_LR*l_CLL)/ms,
                Ks_RF/ms, Cs_RF/ms, Ks_LF/ms, Cs_LF/ms,
                Ks_RR/ms, Cs_RR/ms, Ks_LR/ms, Cs_LR/ms
            ],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [
                (Ks_RF*l_CF+Ks_LF*l_CF-Ks_RR*l_CR-Ks_LR*l_CR)/Iyy,
                (Cs_RF*l_CF+Cs_LF*l_CF-Cs_RR*l_CR-Cs_LR*l_CR)/Iyy,
                (-Ks_RF*l_CF**2-Ks_LF*l_CF**2-Ks_RR*l_CR**2-Ks_LR*l_CR**2)/Iyy,
                (-Cs_RF*l_CF**2-Cs_LF*l_CF**2-Cs_RR*l_CR**2-Cs_LR*l_CR**2)/Iyy,
                (-Ks_RF*l_CF*l_CRR+Ks_LF*l_CF*l_CLL+Ks_RR*l_CR*l_CRR-Ks_LR*l_CR*l_CLL)/Iyy,
                (-Cs_RF*l_CF*l_CRR-Cs_LF*l_CF*l_CLL-Cs_RR*l_CR*l_CRR-Cs_LR*l_CR*l_CLL)/Iyy,
                -Ks_RF*l_CF/Iyy, -Cs_RF*l_CF/Iyy, -Ks_LF*l_CF/Iyy, -Cs_LF*l_CF/Iyy,
                Ks_RR*l_CR/Iyy, Cs_RR*l_CR/Iyy, Ks_LR*l_CR/Iyy, Cs_LR*l_CR/Iyy
            ],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [
                (Ks_RF*l_CRR-Ks_LF*l_CLL+Ks_RR*l_CRR-Ks_LR*l_CLL)/Ixx,
                (Cs_RF*l_CRR-Cs_LF*l_CLL+Cs_RR*l_CRR-Cs_LR*l_CLL)/Ixx,
                (-Ks_RF*l_CF*l_CRR+Ks_LF*l_CF*l_CLL+Ks_RR*l_CR*l_CRR-Ks_LR*l_CR*l_CLL)/Ixx,
                (-Cs_RF*l_CF*l_CRR-Cs_LF*l_CF*l_CLL-Cs_RR*l_CR*l_CRR-Cs_LR*l_CR*l_CLL)/Ixx,
                (-Ks_RF*l_CRR**2-Ks_LF*l_CLL**2-Ks_RR*l_CRR**2-Ks_LR*l_CLL**2)/Ixx,
                (-Cs_RF*l_CRR**2-Cs_LF*l_CLL**2-Cs_RR*l_CRR**2-Cs_LR*l_CLL**2)/Ixx,
                -Ks_RF*l_CRR/Ixx, Cs_RF*l_CRR/Ixx, Ks_LF*l_CLL/Ixx, Cs_LF*l_CLL/Ixx,
                -Ks_RR*l_CRR/Ixx, -Cs_RR*l_CRR/Ixx, Ks_LR*l_CLL/Ixx, Cs_LR*l_CLL/Ixx
            ],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [
                Ks_RF/mw_RF, Cs_RF/mw_RF, -Ks_RF*l_CF/mw_RF, -Cs_RF*l_CF/mw_RF,
                -Ks_RF*l_CRR/mw_RF, -Cs_RF*l_CRR/mw_RF, (-Ks_RF-Kw_RF)/mw_RF, -Cs_RF/mw_RF,
                0, 0, 0, 0, 0, 0
            ],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [
                Ks_LF/mw_LF, Cs_LF/mw_LF, -Ks_LF*l_CF/mw_LF, -Cs_LF*l_CF/mw_LF,
                Ks_LF*l_CLL/mw_LF, Cs_LF*l_CLL/mw_LF, 0, 0,
                (-Ks_LF-Kw_LF)/mw_LF, -Cs_LF/mw_LF, 0, 0, 0, 0
            ],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [
                Ks_RR/mw_RR, Cs_RR/mw_RR, Ks_RR*l_CR/mw_RR, Cs_RR*l_CR/mw_RR,
                Ks_RR*l_CLL/mw_RR, Cs_RR*l_CLL/mw_RR, 0, 0, 0, 0,
                (-Ks_RR-Kw_RR)/mw_RR, -Cs_RR/mw_RR, 0, 0
            ],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [
                Ks_LR/mw_LR, Cs_LR/mw_LR, Ks_LR*l_CR/mw_LR, Cs_LR*l_CR/mw_LR,
                Ks_LR*l_CLL/mw_LR, Cs_LR*l_CLL/mw_LR, 0, 0, 0, 0, 0, 0,
                (-Ks_LR-Kw_LR)/mw_LR, -Cs_LR/mw_LR
            ]
        ])

        I4 = np.eye(4)
        B = np.vstack(
            (
                np.zeros((self.n_control)),
                1/ms * np.ones((self.n_control)),
                np.zeros((self.n_control)),
                np.array([-l_CF/Iyy, -l_CF/Iyy, l_CR/Iyy, l_CR/Iyy]),
                np.zeros((self.n_control)),
                np.array([l_CRR/Ixx, -l_CLL/Ixx, l_CRR/Ixx, l_CLL/Ixx]),
                np.zeros((self.n_control)),
                -1/ms * I4[0, :],
                np.zeros((self.n_control)),
                -1/ms * I4[1, :],
                np.zeros((self.n_control)),
                -1/ms * I4[2, :],
                np.zeros((self.n_control)),
                -1/ms * I4[3, :]
            )
        )
        sysd = cont2discrete(
            (A, B, np.eye(self.n_states), np.zeros_like(B)),
            self.Ts
        )

        self.Ad = ca.reshape(sysd[0], self.n_states, self.n_states)
        self.Bd = ca.reshape(sysd[1], self.n_states, self.n_control)

    def cal(self, x, u, w=None):
        if w is None:
            x_next = ca.mtimes(self.Ad, x) + ca.mtimes(self.Bd, u)
        else:
            x_next = ca.mtimes(self.Ad, x) + ca.mtimes(ca.mtimes(self.Bd, u), w)

        #[z dz phi dphi theta dtheta z_LF dz_LF z_RF dz_RF z_LR dz_LR z_RR dz_RR]
        return x_next

    def passive(self, x):
        L = self.L
        W = self.W

        delta_fl_dot = W/2*x[3]*ca.cos(x[2]) - L/2*x[5]*ca.cos(x[4]) + x[1] - x[7]
        delta_fr_dot = -W/2*x[3]*ca.cos(x[2]) - L/2*x[5]*ca.cos(x[4]) + x[1] - x[9]
        delta_rl_dot = W/2*x[3]*ca.cos(x[2]) + L/2*x[5]*ca.cos(x[4]) + x[1] - x[11]
        delta_rr_dot = -W/2*x[3]*ca.cos(x[2]) + L/2*x[5]*ca.cos(x[4]) + x[1] - x[13]

        u_fl = 1000 * delta_fl_dot
        u_fr = 1000 * delta_fr_dot
        u_rl = 1000 * delta_rl_dot
        u_rr = 1000 * delta_rr_dot

        return [u_fl,u_fr,u_rl,u_rr]
