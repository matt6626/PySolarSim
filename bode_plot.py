import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import mplcursors
import numpy as np
import plot_helper as ph

class bode_plot_zpk:
    def __init__(
        self, G0: float = 1, f_zeroes: list[float] = [], f_poles: list[float] = []
    ) -> None:
        self.G0 = np.array(G0, dtype=complex)
        self.f_zeroes = sorted(np.array(f_zeroes, dtype=complex))
        self.f_poles = sorted(np.array(f_poles, dtype=complex))
        return

    def plot(self, fstart, fstop, linear=False):
        fsteps = 100
        if linear:
            f = np.linspace(fstart, fstop, fsteps)
        else:
            fstart = np.log10(fstart)
            fstop = np.log10(fstop)
            f = np.logspace(fstart, fstop, fsteps)
        flength = len(f)

        # Note: Decibel gain is normalised to value of 1 for any quantity that is not dimensionless
        tf_mag_db: list[ph.plot_data] = []
        tf_phase_deg: list[ph.plot_data] = []

        # DC gain
        G0 = self.G0
        mag = G0 * np.ones(flength)
        db = 20 * np.log10(mag)
        tf_mag_db.append(ph.plot_data(db, "DC Gain"))
        phase_deg = np.zeros(flength)
        tf_phase_deg.append(ph.plot_data(phase_deg, "DC Phase"))

        # Zeroes
        f_zeroes = [zero for zero in self.f_zeroes if zero != 0]
        f_origin_zeros = [zero for zero in self.f_zeroes if zero == 0]

        # handle origin zeroes
        for i, f_zero in enumerate(f_origin_zeros):
            if f_zero == 0:
                g_jw = f * 1j
                mag = np.abs(g_jw)
                db = 20 * np.log10(mag)
                tf_mag_db.append(ph.plot_data(db, f"Zero (f = {0} Hz)"))
                phase_deg = np.rad2deg(np.angle(g_jw))
                tf_phase_deg.append(ph.plot_data(phase_deg, f"Zero (f = {0} Hz)"))

        # G = (1 + j*f/f0)
        # |G| = sqrt([1]^2 + (f/f0)^2)
        # |G|_db = 20 * log10(|G|)
        for f_zero in f_zeroes:
            f_mag = np.abs(f_zero)
            g_jw = 1 + (f / f_zero) * 1j
            mag = np.abs(g_jw)
            db = 20 * np.log10(mag)
            tf_mag_db.append(ph.plot_data(db, f"Zero (f = {f_mag} Hz)"))
            phase_deg = np.rad2deg(np.angle(g_jw))
            tf_phase_deg.append(ph.plot_data(phase_deg, f"Zero (f = {f_mag} Hz)"))

        # Poles
        f_poles = [pole for pole in self.f_poles if pole != 0]
        f_origin_poles = [pole for pole in self.f_poles if pole == 0]

        # handle origin poles
        for i, f_pole in enumerate(f_origin_poles):
            g_jw = 1 / (f * 1j)
            mag = np.abs(g_jw)
            db = 20 * np.log10(mag)
            tf_mag_db.append(ph.plot_data(db, f"Pole (f = {0} Hz)"))
            phase_deg = np.rad2deg(np.angle(g_jw))
            tf_phase_deg.append(ph.plot_data(phase_deg, f"Pole (f = {0} Hz)"))

        # G = 1 / (1 + j*f/f0)
        # |G| = 1 / sqrt([1]^2 + (f/f0)^2)
        # TODO: revert this to zero form and just flip
        # |G|_db = 20 * log10(|G|)
        for f_pole in f_poles:
            f_mag = np.abs(f_pole)
            g_jw = 1 / (1 + (f / f_pole) * 1j)
            mag = np.abs(g_jw)
            db = 20 * np.log10(mag)
            tf_mag_db.append(ph.plot_data(db, f"Pole (f = {f_mag:.2f} Hz)"))
            phase_deg = np.rad2deg(np.angle(g_jw))
            tf_phase_deg.append(ph.plot_data(phase_deg, f"Pole (f = {f_mag:.2f} Hz)"))

        # Overall TF
        tot_mag = np.zeros(flength)
        tot_phase_deg = np.zeros(flength)
        for pd in tf_mag_db:
            tot_mag = tot_mag + pd.ydata
        for pd in tf_phase_deg:
            tot_phase_deg = tot_phase_deg + pd.ydata
        tf_mag_db.append(ph.plot_data(tot_mag, f"Combined Magnitude (dB)"))
        tf_phase_deg.append(ph.plot_data(tot_phase_deg, "Combined Phase (deg)"))

        nplots = len(tf_mag_db)
        fig = plt.figure(constrained_layout=True)
        axs: list[Axes] = fig.subplots(2, nplots)
        for i in range(nplots):

            # Magnitude
            mag_db = tf_mag_db[i]
            mag_ax = axs[0][i]
            mag_ydata = mag_db.ydata
            mag_ylabel = mag_db.ylabel
            if linear:
                mag_ax.plot(f, mag_ydata)
            else:
                mag_ax.semilogx(f, mag_ydata)
            mag_ax.set_ylabel(mag_ylabel)
            mag_ax.set_xlabel("Frequency (Hz)")
            mag_ax.grid()

            # Phase
            phase_deg = tf_phase_deg[i]
            phase_ax = axs[1][i]
            phase_ydata = phase_deg.ydata
            phase_ylabel = phase_deg.ylabel
            if linear:
                phase_ax.plot(f, phase_ydata)
            else:
                phase_ax.semilogx(f, phase_ydata)
            phase_ax.set_ylabel(phase_ylabel)
            phase_ax.set_xlabel("Frequency (Hz)")
            phase_ax.grid()

        return fig

    # takes real poles/zeros and uses the easy to read equations
    # TODO: deprecrate in favour of the more generic plot which takes complex poles/zeroes
    def plot_real(self, fstart, fstop, linear=False):
        fsteps = 100
        if linear:
            f = np.linspace(fstart, fstop, fsteps)
        else:
            fstart = np.log10(fstart)
            fstop = np.log10(fstop)
            f = np.logspace(fstart, fstop, fsteps)
        flength = len(f)

        # Note: Decibel gain is normalised to value of 1 for any quantity that is not dimensionless
        tf_mag_db: list[ph.plot_data] = []
        tf_phase_deg: list[ph.plot_data] = []

        # DC gain
        G0 = self.G0
        mag = G0 * np.ones(flength)
        db = 20 * np.log10(mag)
        tf_mag_db.append(ph.plot_data(db, "DC Gain"))
        phase_deg = np.zeros(flength)
        tf_phase_deg.append(ph.plot_data(phase_deg, "DC Phase"))

        # Zeroes
        # |G| = (1+(f/f0)^2)^-0.5
        # |G|_db = 20*log10(f/f0)
        f_zeroes = self.f_zeroes
        for f_zero in f_zeroes:
            mag = (1 + (f / f_zero) ** 2) ** -0.5
            db = 20 * np.log10(mag)
            tf_mag_db.append(ph.plot_data(db, f"Zero (f = {f_zero} Hz)"))
            phase_deg = np.rad2deg(np.arctan(f / f_zero))
            tf_phase_deg.append(ph.plot_data(phase_deg, f"Zero (f = {f_zero} Hz)"))

        # Poles
        # |G| = (1+(f/f0)^2)^0.5
        # |G|_db = -20*log10(f/f0)
        f_poles = self.f_poles
        for f_pole in f_poles:
            mag = (1 + (f / f_pole) ** 2) ** -0.5
            db = 20 * np.log10(mag)
            tf_mag_db.append(ph.plot_data(db, f"Zero (f = {f_pole} Hz)"))
            phase_deg = np.rad2deg(-np.arctan(f / f_pole))
            tf_phase_deg.append(ph.plot_data(phase_deg, f"Zero (f = {f_pole} Hz)"))

        # Overall TF
        tot_mag = np.zeros(flength)
        tot_phase_deg = np.zeros(flength)
        for pd in tf_mag_db:
            tot_mag = tot_mag + pd.ydata
        for pd in tf_phase_deg:
            tot_phase_deg = tot_phase_deg + pd.ydata
        tf_mag_db.append(ph.plot_data(tot_mag, f"Combined Magnitude (dB)"))
        tf_phase_deg.append(ph.plot_data(tot_phase_deg, "Combined Phase (deg)"))

        nplots = len(tf_mag_db)
        fig = plt.figure(constrained_layout=True)
        axs: list[Axes] = fig.subplots(2, nplots)
        for i in range(nplots):

            # Magnitude
            mag_db = tf_mag_db[i]
            mag_ax = axs[0][i]
            mag_ydata = mag_db.ydata
            mag_ylabel = mag_db.ylabel
            if linear:
                mag_ax.plot(f, mag_ydata)
            else:
                mag_ax.semilogx(f, mag_ydata)
            mag_ax.set_ylabel(mag_ylabel)
            mag_ax.set_xlabel("Frequency (Hz)")
            mag_ax.grid()

            # Phase
            phase_deg = tf_phase_deg[i]
            phase_ax = axs[1][i]
            phase_ydata = phase_deg.ydata
            phase_ylabel = phase_deg.ylabel
            if linear:
                phase_ax.plot(f, phase_ydata)
            else:
                phase_ax.semilogx(f, phase_ydata)
            phase_ax.set_ylabel(phase_ylabel)
            phase_ax.set_xlabel("Frequency (Hz)")
            phase_ax.grid()
        return fig

    def example1(self):
        # To run example use:
        # bp = bode_plot_zpk()
        # bp.example1()
        self.G0 = 40
        self.f_poles = [100, 2000]

        fig = self.plot(1, 10e3)
        plt.show()

        return

    def example2(self):
        # To run example use:
        # bp = bode_plot_zpk()
        # bp.example2()
        self.G0 = 1
        self.f_zeroes = [0]
        self.f_poles = [0, 0]

        fig = self.plot(1, 10e3)
        plt.show()

        return
