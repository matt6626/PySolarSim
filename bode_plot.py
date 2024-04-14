import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import mplcursors
import numpy as np


class plot_data:
    def __init__(self, ydata, ylabel) -> None:
        self.ydata = ydata
        self.ylabel = ylabel
        return


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
        tf_mag_db: list[plot_data] = []
        tf_phase_deg: list[plot_data] = []

        # DC gain
        G0 = self.G0
        mag = G0 * np.ones(flength)
        db = 20 * np.log10(mag)
        tf_mag_db.append(plot_data(db, "DC Gain"))
        phase_deg = np.zeros(flength)
        tf_phase_deg.append(plot_data(phase_deg, "DC Phase"))

        # Zeroes
        # G = (1 + j*f/f0)
        # |G| = sqrt([1]^2 + (f/f0)^2)
        # |G|_db = 20 * log10(|G|)
        f_zeroes = self.f_zeroes
        for f_zero in f_zeroes:
            f_mag = np.abs(f_zero)
            g_jw = 1 + (f / f_zero) * 1j
            mag = np.abs(g_jw)
            db = 20 * np.log10(mag)
            tf_mag_db.append(plot_data(db, f"Zero (f = {f_mag} Hz)"))
            phase_deg = np.rad2deg(np.angle(g_jw))
            tf_phase_deg.append(plot_data(phase_deg, f"Zero (f = {f_mag} Hz)"))

        # Poles
        # G = (1 - j*f/f0) / [1 + (f/f0)^2]
        # |G| = sqrt([1/(1 + [f/f0]^2)]^2 + [(-f/f0)/)(1+[f/f0]^2)]^2)
        # TODO: revert this to zero form and just flip
        # |G|_db = 20 * log10(|G|)
        f_poles = self.f_poles
        for f_pole in f_poles:
            f_mag = np.abs(f_pole)
            g_jw = 1 / (1 + (f / f_pole) * 1j)
            mag = np.abs(g_jw)
            db = 20 * np.log10(mag)
            tf_mag_db.append(plot_data(db, f"Pole (f = {f_mag:.2f} Hz)"))
            phase_deg = np.rad2deg(np.angle(g_jw))
            tf_phase_deg.append(plot_data(phase_deg, f"Pole (f = {f_mag:.2f} Hz)"))

        # Overall TF
        tot_mag = np.zeros(flength)
        tot_phase_deg = np.zeros(flength)
        for pd in tf_mag_db:
            tot_mag = tot_mag + pd.ydata
        for pd in tf_phase_deg:
            tot_phase_deg = tot_phase_deg + pd.ydata
        tf_mag_db.append(plot_data(tot_mag, f"Combined Magnitude (dB)"))
        tf_phase_deg.append(plot_data(tot_phase_deg, "Combined Phase (deg)"))

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
        tf_mag_db: list[plot_data] = []
        tf_phase_deg: list[plot_data] = []

        # DC gain
        G0 = self.G0
        mag = G0 * np.ones(flength)
        db = 20 * np.log10(mag)
        tf_mag_db.append(plot_data(db, "DC Gain"))
        phase_deg = np.zeros(flength)
        tf_phase_deg.append(plot_data(phase_deg, "DC Phase"))

        # Zeroes
        # |G| = (1+(f/f0)^2)^-0.5
        # |G|_db = 20*log10(f/f0)
        f_zeroes = self.f_zeroes
        for f_zero in f_zeroes:
            mag = (1 + (f / f_zero) ** 2) ** -0.5
            db = 20 * np.log10(mag)
            tf_mag_db.append(plot_data(db, f"Zero (f = {f_zero} Hz)"))
            phase_deg = np.rad2deg(np.arctan(f / f_zero))
            tf_phase_deg.append(plot_data(phase_deg, f"Zero (f = {f_zero} Hz)"))

        # Poles
        # |G| = (1+(f/f0)^2)^0.5
        # |G|_db = -20*log10(f/f0)
        f_poles = self.f_poles
        for f_pole in f_poles:
            mag = (1 + (f / f_pole) ** 2) ** -0.5
            db = 20 * np.log10(mag)
            tf_mag_db.append(plot_data(db, f"Zero (f = {f_pole} Hz)"))
            phase_deg = np.rad2deg(-np.arctan(f / f_pole))
            tf_phase_deg.append(plot_data(phase_deg, f"Zero (f = {f_pole} Hz)"))

        # Overall TF
        tot_mag = np.zeros(flength)
        tot_phase_deg = np.zeros(flength)
        for pd in tf_mag_db:
            tot_mag = tot_mag + pd.ydata
        for pd in tf_phase_deg:
            tot_phase_deg = tot_phase_deg + pd.ydata
        tf_mag_db.append(plot_data(tot_mag, f"Combined Magnitude (dB)"))
        tf_phase_deg.append(plot_data(tot_phase_deg, "Combined Phase (deg)"))

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

    def example(self):
        # To run example use:
        # bp = bode_plot_zpk()
        # bp.example()
        self.G0 = 40
        self.f_poles = [100, 2000]

        fig = self.plot(1, 10e3)
        plt.show()

        return
