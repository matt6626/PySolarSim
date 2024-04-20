import numpy as np
from collections import namedtuple
import plot_helper as ph

class voltage_mode_controller:

    def __init__(
        self,
    ) -> None:
        self.simulation_not_started = True
        self.v_ref = 0
        self.v_out = 0
        self.v_err = 0
        self.v_control = 0

    def control_func(self, reference, input, dt):
        return input

    def simulate(self, v_ref, v_out, dt, plot=False):
        if self.simulation_not_started:
            if plot:
                self.plot(dt, init=True)
            self.simulation_not_started = False
        v_control = self.control_func(v_ref, v_out, dt)
        self.v_ref = v_ref
        self.v_out = v_out
        self.v_err = v_ref - v_out
        self.v_control = v_control

        if plot:
            self.plot(dt)
        return v_control

    def get_controller_state_plot_data(self) -> list[ph.plot_data]:
        plot_data: list[ph.plot_data] = []
        plot_data.append(ph.plot_data(self.v_ref, "v_ref"))
        plot_data.append(ph.plot_data(self.v_out, "v_out"))
        plot_data.append(ph.plot_data(self.v_err, "v_err"))
        return plot_data

    def plot(self, dt, init=False):
        import matplotlib.pyplot as plt
        import numpy as np

        if init:
            self.vars = dict()
            plot_data_list = self.get_controller_state_plot_data()
            for plot_data in plot_data_list:
                key = plot_data.ylabel
                value = plot_data.ydata
                self.vars[key] = [value]

            self.t = [0]
            plt.ion()
            self.fig, self.ax = plt.subplots(len(self.vars), 1)
            self.lines = {}

            for i, (key, value) in enumerate(self.vars.items()):
                (line,) = self.ax[i].plot(self.t, value, animated=True)
                self.ax[i].set(xlabel="time (s)", ylabel=f"{key}")
                self.lines[key] = line

            self.fig.canvas.draw()
            self.bg = [self.fig.canvas.copy_from_bbox(ax.bbox) for ax in self.ax]

        else:
            self.t.append(self.t[-1] + dt)
            plot_data_list = self.get_controller_state_plot_data()

            for plot_data in plot_data_list:
                key = plot_data.ylabel
                value = plot_data.ydata
                self.vars[key].append(value)
                self.lines[key].set_ydata(self.vars[key])
                self.lines[key].set_xdata(self.t)

            for i, (key, ax) in enumerate(zip(self.vars.keys(), self.ax)):
                ax.relim()
                ax.autoscale_view()
                self.fig.canvas.restore_region(self.bg[i])
                ax.draw_artist(self.lines[key])

            # Blit all subplots at once
            bbox = self.fig.bbox.union([ax.bbox for ax in self.ax])
            self.fig.canvas.blit(bbox)
            # self.fig.canvas.draw_idle()
            plt.pause(0.001)

    def bode(self):
        import bode_plot as bp


class analog_type_1_with_dc_gain_controller(voltage_mode_controller):
    # type 1 compensator circuit
    #                          ___      ||
    #                      ---|___|-----||----
    #                      |            ||   |
    #                      |   Rf       Cf   |
    #               Rg     |                 |
    #               ___    |       |\|       |
    #    v_out o----|___|---o-------|-\       |
    #                              |  >------o----o v_control
    #    v_ref o--------------------|+/
    #                              |/|

    # This named tuple might be a nicer approach to initialisation compared to passing many parameters
    # TODO: investigate this
    control_func_params = namedtuple(
        "control_func_params",
        ["v_cf_0", "rg", "rf", "cf", "vsupply_neg", "vsupply_pos"],
    )

    def __init__(
        self,
        rg=1e3,
        rf=0,
        cf=1e-9,
        vsupply_neg=-np.Inf,
        vsupply_pos=np.Inf,
        v_cf_0=0,
        v_control_0=0,
    ) -> None:

        super().__init__()

        # Controller parameters
        self.rg = rg
        self.rf = rf
        self.cf = cf
        self.vsupply_neg = vsupply_neg
        self.vsupply_pos = vsupply_pos

        # State variables [memory]
        self.v_cf = v_cf_0
        self.i_cf = 0
        self.v_control = v_control_0

    def get_controller_state_plot_data(self) -> list[ph.plot_data]:
        plot_data: list[ph.plot_data] = []
        plot_data.append(ph.plot_data(self.v_ref, "v_ref"))
        plot_data.append(ph.plot_data(self.v_out, "v_out"))
        plot_data.append(ph.plot_data(self.v_err, "v_err"))
        plot_data.append(ph.plot_data(self.v_control, "v_control"))
        plot_data.append(ph.plot_data(self.v_cf, "v_cf"))
        plot_data.append(ph.plot_data(self.i_cf, "i_cf"))
        return plot_data

    def control_func(self, reference, input, dt):

        v_ref = reference
        v_out = input
        v_cf_0 = self.v_cf
        rg = self.rg
        rf = self.rf
        cf = self.cf
        vsupply_neg = self.vsupply_neg
        vsupply_pos = self.vsupply_pos

        i_cf = (v_ref - v_out) / rg

        v_cf = ((i_cf / cf) * dt) + v_cf_0

        v_control = np.clip(
            v_cf + (rf * i_cf) + v_ref,
            vsupply_neg,
            vsupply_pos,
        )

        # If saturation occurs, recalculate v_cf and i_cf
        # v_out + i_cf * (rf + rg) + v_cf = v_control
        # v_cf = v_control - i_cf * (rf + rg) - v_out
        # i_cf = C dv_cf / dt
        #
        # i_cf = C * (v_cf - v_cf_0) / dt
        # i_cf = C * (v_control - i_cf * (rf + rg) - v_out - v_cf_0) / dt
        #
        # i_cf * dt = C * (v_control - i_cf * (rf + rg) - v_out - v_cf_0)
        # i_cf * dt + i_cf * C * (rf + rg)  = C * (v_control - v_out - v_cf_0)
        # i_cf * (dt + C * (rf + rg))  = C * (v_control - v_out - v_cf_0)
        # i_cf = C * (v_control - v_out - v_cf_0) / (dt + C * (rf + rg))
        if v_control == vsupply_neg or v_control == vsupply_pos:
            i_cf = cf * (v_control - v_out - v_cf_0) / (dt + cf * (rf + rg))
            v_cf = ((i_cf / cf) * dt) + v_cf_0

        # assign state variables
        self.v_cf = v_cf
        self.i_cf = i_cf

        return v_control


class analog_type3_compensator_controller(voltage_mode_controller):
    # type 3 compensator circuit
    # https://www.ti.com/lit/an/slva662/slva662.pdf

    def __init__(
        self,
        r1=1e3,
        r2=1e3,
        r3=1e3,
        r4=1e3,
        c1=1e-9,
        c2=1e-9,
        c3=1e-9,
        vsupply_neg=-np.Inf,
        vsupply_pos=np.Inf,
        open_loop_gain=10e6,
        v_c1_0=0,
        v_c2_0=0,
        v_c3_0=0,
        v_control_0=0,
    ) -> None:

        super().__init__()

        # Controller parameters
        self.r1 = r1
        self.r2 = r2
        self.r3 = r3
        self.r4 = r4
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.vsupply_neg = vsupply_neg
        self.vsupply_pos = vsupply_pos
        self.open_loop_gain = open_loop_gain

        # State variables [memory]
        self.v_c1 = v_c1_0
        self.v_c2 = v_c2_0
        self.v_c3 = v_c3_0
        self.v_control = v_control_0

    def get_controller_state_plot_data(self) -> list[ph.plot_data]:
        plot_data: list[ph.plot_data] = []
        plot_data.append(ph.plot_data(self.v_ref, "v_ref"))
        plot_data.append(ph.plot_data(self.v_out, "v_out"))
        plot_data.append(ph.plot_data(self.v_err, "v_err"))
        plot_data.append(ph.plot_data(self.v_control, "v_control"))
        return plot_data

    def control_func(self, reference, input, dt):

        v_ref = reference
        v_out = input
        v_c1_0 = self.v_c1
        v_c2_0 = self.v_c2
        v_c3_0 = self.v_c3
        r1 = self.r1
        r2 = self.r2
        r3 = self.r3
        r4 = self.r4
        c1 = self.c1
        c2 = self.c2
        c3 = self.c3
        vsupply_neg = self.vsupply_neg
        vsupply_pos = self.vsupply_pos
        open_loop_gain = self.open_loop_gain

        if v_ref < vsupply_neg:
            raise Exception(
                f"v_ref violates saturation limits (v_ref={v_ref}) < vsupply_neg={vsupply_neg}"
            )
        if v_ref > vsupply_pos:
            raise Exception(
                f"v_ref violates saturation limits (v_ref={v_ref}) > vsupply_pos={vsupply_pos}"
            )

        # Zi
        i_r1 = (v_ref - v_out) / r1
        # v_c2 eqn derived from:
        # eqn1: Ic = C (Vc - Vc0) / dt
        # eqn2: Vc = v_ref - v_out - Ic * R3
        # eqn1 -> eqn2: Vc = v_ref - v_out - C * R3 * (Vc - Vc0) / dt
        v_c2 = (v_ref - v_out + (r3 * c2 * v_c2_0 / dt)) / (1 + (r3 * c2 / dt))
        i_c2 = c2 * (v_c2 - v_c2_0) / dt
        i_zi = i_r1 + i_c2

        # Zf
        # equations:
        # eqn1: v_c3 = v_control - v_ref
        # eqn2: i_c3 = c3 * (v_c3 - v_c3_0) / dt
        # eqn1 => eqn2 (eqn3): i_c3 = c3 * (v_control - v_ref - v_c3_0) / dt
        # eqn4: i_c3 + i_c1 = i_zi + v_ref / r4
        # eqn4 => eqn3 (eqn5): i_zi + v_ref / r4 - i_c1 = c3 * (v_control - v_ref - v_c3_0) / dt
        # eqn6: i_c1 = c1 * (v_c1 - v_c1_0) / dt
        # eqn6 => eqn5 (eqn7): i_zi + v_ref / r4 - c1 * (v_c1 - v_c1_0) / dt = c3 * (v_control - v_ref - v_c3_0) / dt
        # rearrange eqn7 (eqn8***): i_zi * dt + v_ref / r4 * dt - c1 * (v_c1 - v_c1_0) = c3 * (v_control - v_ref - v_c3_0)
        # eqn9: v_c1 = v_control - v_ref - i_c1 * r2
        # eqn6 => eqn9 (eqn10***): v_c1 = v_control - v_ref - r2 * c1 * (v_c1 - v_c1_0) / dt
        # use eqn8, eqn10 to solve for v_c1 and v_control:
        # rearrange eqn8 (eqn11): v_control = (i_zi * dt + v_ref / r4 * dt - c1 * (v_c1 - v_c1_0)) / c3 + v_ref + v_c3_0
        # eqn11 => eqn10 (eqn12): v_c1 + c1 * v_c1 / c3 + r2 * c1 * v_c1 / dt = i_zi * dt / c3 + v_ref * dt / (r4 * c3) + c1 * v_c1_0 / c3 + v_c3_0 + r2 * c1 * v_c1_0 / dt
        # simplify eqn12 (eqn13): v_c1 = (i_zi * dt / c3 + v_ref * dt / (r4 * c3) + c1 * v_c1_0 / c3 + v_c3_0 + r2 * c1 * v_c1_0 / dt) / (1 + c1 / c3 + r2 * c1 / dt)
        # then arrange eqn10 to solve for v_control (eqn14): v_control = v_c1 + v_ref + r2 * c1 * (v_c1 - v_c1_0) / dt
        v_c1 = (
            i_zi * dt / c3
            + v_ref * dt / (r4 * c3)
            + c1 * v_c1_0 / c3
            + v_c3_0
            + r2 * c1 * v_c1_0 / dt
        ) / (1 + c1 / c3 + r2 * c1 / dt)
        v_control = np.clip(
            v_c1 + v_ref + r2 * c1 * (v_c1 - v_c1_0) / dt,
            vsupply_neg,
            vsupply_pos,
        )

        v_c3 = v_control - v_ref
        i_c3 = c3 * (v_c3 - v_c3_0) / dt
        i_c1 = i_zi + v_ref / r4 - i_c3
        v_in_neg = v_control - v_c3

        # opamp saturation
        # if saturation occurred, calculate the true voltage at the inverting terminal, and recalculate circuit voltages/currents as result
        # should be able to always perform this calculation
        if v_control == vsupply_neg or v_control == vsupply_pos:
            # Saturation equations

            # equation relating Zf and Zi:
            # i_zi = i_r1 + i_c2
            # i_zf = i_c1 + i_c3
            # v_in_neg = (i_zf - i_zi)* r4
            # eqn (*): v_in_neg = (i_c1 + i_c3 - i_r1 - i_c2)* r4

            # equations for Zi:
            # v_c2 = (v_in_neg - v_out + (r3 * c2 * v_c2_0 / dt) / (1 + (r3 * c2 / dt))
            # i_c2 = c2 * (v_c2 - v_c2_0) / dt
            # i_r1 = (v_in_neg - v_out) / r1
            # i_zi = i_r1 + i_c2

            # substitute (*) => Zi equations:
            # v_c2 = [( i_c1 + i_c3 - i_r1 - i_c2)* r4 - v_out + (r3 * c2 * v_c2_0 / dt)] / [1 + (r3 * c2 / dt)]

            # equations for Zf:
            # v_c1 = [i_zi * dt / c3 + v_ref * dt / (r4 * c3) + c1 * v_c1_0 / c3 + v_c3_0 + r2 * c1 * v_c1_0 / dt] / [1 + c1 / c3 + r2 * c1 / dt]
            # v_c3 = v_control - v_in_neg
            # i_c3 = c3 * (v_c3 - v_c3_0) / dt
            # i_c1 = c1 * (v_c1 - v_c1_0) / dt
            # i_zf = i_c1 + i_c3

            # substitute (*) => Zf equations:
            # v_c1 = [(i_r1 + i_c2) * dt / c3 + v_ref * dt / (r4 * c3) + c1 * v_c1_0 / c3 + v_c3_0 + r2 * c1 * v_c1_0 / dt] / [1 + c1 / c3 + r2 * c1 / dt]
            # v_c3 = v_control - ( i_c1 + i_c3 - i_r1 - i_c2)* r4

            # list all our equations:
            # (1):
            # v_c1 = [(i_r1 + i_c2) * dt / c3 + v_ref * dt / (r4 * c3) + c1 * v_c1_0 / c3 + v_c3_0 + r2 * c1 * v_c1_0 / dt] / [1 + c1 / c3 + r2 * c1 / dt]

            # (2):
            # i_c1 = c1 * (v_c1 - v_c1_0) / dt

            # (3):
            # v_c2 = [( i_c1 + i_c3 - i_r1 - i_c2)* r4 - v_out + (r3 * c2 * v_c2_0 / dt)] / [1 + (r3 * c2 / dt)]

            # (4):
            # i_c2 = c2 * (v_c2 - v_c2_0) / dt

            # (5):
            # v_c3 = v_control - ( i_c1 + i_c3 - i_r1 - i_c2)* r4

            # (6):
            # i_c3 = c3 * (v_c3 - v_c3_0) / dt

            # (7):
            # i_r1 = (v_in_neg - v_out) / r1
            # i_r1 = [(i_c1 + i_c3 - i_r1 - i_c2)* r4 - v_out] / r1
            # i_r1 * r1 = (i_c1 + i_c3 - i_r1 - i_c2)* r4 - v_out
            # i_r1 * r1 + i_r1 * r4 = (i_c1 + i_c3 - i_c2)* r4 - v_out
            # i_r1 * (r1 + r4) = (i_c1 + i_c3 - i_c2)* r4 - v_out
            # i_r1 = [(i_c1 + i_c3 - i_c2)* r4 - v_out] / [r1 + r4]
            # sub (2), (4), (6):
            # i_r1 = ([(c1 * (v_c1 - v_c1_0) / dt + c3 * (v_c3 - v_c3_0) / dt - c2 * (v_c2 - v_c2_0) / dt)* r4 - v_out] / [r1 + r4])

            # sub (2), (4), (6), (7) into (1), (3), (5):
            # (8):
            # v_c1 = [(([(c1 * (v_c1 - v_c1_0) / dt + c3 * (v_c3 - v_c3_0) / dt - c2 * (v_c2 - v_c2_0) / dt)* r4 - v_out] / [r1 + r4]) + c2 * (v_c2 - v_c2_0) / dt) * dt / c3 + v_ref * dt / (r4 * c3) + c1 * v_c1_0 / c3 + v_c3_0 + r2 * c1 * v_c1_0 / dt] / [1 + c1 / c3 + r2 * c1 / dt]

            # (9):
            # v_c2 = [c1 * (v_c1 - v_c1_0) / dt + c3 * (v_c3 - v_c3_0) / dt - i_r1 - c2 * (v_c2 - v_c2_0) / dt)* r4 - v_out + (r3 * c2 * v_c2_0 / dt)] / [1 + (r3 * c2 / dt)]

            # (10):
            # v_c3 = v_control - (c1 * (v_c1 - v_c1_0) / dt + c3 * (v_c3 - v_c3_0) / dt - ([(c1 * (v_c1 - v_c1_0) / dt + c3 * (v_c3 - v_c3_0) / dt - c2 * (v_c2 - v_c2_0) / dt)* r4 - v_out] / [r1 + r4]) - c2 * (v_c2 - v_c2_0) / dt)* r4

            # if you do all the maths, then use sympy to solve the simulatenous equations
            def calc_vc1(
                c1,
                c2,
                c3,
                dt,
                r1,
                r2,
                r3,
                r4,
                v_c1_0,
                v_c2_0,
                v_c3_0,
                v_control,
                v_out,
                v_ref,
            ):
                return (
                    c1 * c2 * (c3**2) * r1 * r2 * r3 * (r4**2) * v_c1_0
                    + c1 * c2 * c3 * dt * r1 * r2 * r3 * r4 * v_c1_0
                    + c1 * c2 * c3 * dt * r1 * r2 * (r4**2) * v_c1_0
                    + c1 * c2 * c3 * dt * r1 * r3 * (r4**2) * v_c1_0
                    + c1 * c2 * c3 * dt * r2 * r3 * (r4**2) * v_c1_0
                    + c1 * c2 * (dt**2) * r1 * r3 * r4 * v_c1_0
                    + c1 * (c3**2) * dt * r1 * r2 * (r4**2) * v_c1_0
                    + c1 * c3 * (dt**2) * r1 * r2 * r4 * v_c1_0
                    + c1 * c3 * (dt**2) * r1 * (r4**2) * v_c1_0
                    + c1 * c3 * (dt**2) * r2 * (r4**2) * v_c1_0
                    + c1 * (dt**3) * r1 * r4 * v_c1_0
                    + c2 * (c3**2) * dt * r1 * r3 * (r4**2) * v_c3_0
                    + c2 * c3 * (dt**2) * r1 * r3 * r4 * v_c3_0
                    + c2 * c3 * (dt**2) * r1 * r3 * r4 * v_ref
                    - c2 * c3 * (dt**2) * r1 * (r4**2) * v_c2_0
                    + c2 * c3 * (dt**2) * r1 * (r4**2) * v_control
                    - c2 * c3 * (dt**2) * r1 * (r4**2) * v_out
                    + c2 * c3 * (dt**2) * r3 * (r4**2) * v_control
                    - c2 * c3 * (dt**2) * r3 * (r4**2) * v_out
                    + c2 * (dt**3) * r1 * r3 * v_ref
                    - c2 * (dt**3) * r1 * r4 * v_c2_0
                    - c2 * (dt**3) * r1 * r4 * v_out
                    + c2 * (dt**3) * r1 * r4 * v_ref
                    - c2 * (dt**3) * r3 * r4 * v_out
                    + c2 * (dt**3) * r3 * r4 * v_ref
                    + (c3**2) * (dt**2) * r1 * (r4**2) * v_c3_0
                    + c3 * (dt**3) * r1 * r4 * v_c3_0
                    + c3 * (dt**3) * r1 * r4 * v_ref
                    + c3 * (dt**3) * (r4**2) * v_control
                    - c3 * (dt**3) * (r4**2) * v_out
                    + (dt**4) * r1 * v_ref
                    - (dt**4) * r4 * v_out
                    + (dt**4) * r4 * v_ref
                ) / (
                    r4
                    * (
                        c1 * c2 * (c3**2) * r1 * r2 * r3 * r4
                        + c1 * c2 * c3 * dt * r1 * r2 * r3
                        + c1 * c2 * c3 * dt * r1 * r2 * r4
                        + c1 * c2 * c3 * dt * r1 * r3 * r4
                        + c1 * c2 * c3 * dt * r2 * r3 * r4
                        + c1 * c2 * (dt**2) * r1 * r3
                        + c1 * (c3**2) * dt * r1 * r2 * r4
                        + c1 * c3 * (dt**2) * r1 * r2
                        + c1 * c3 * (dt**2) * r1 * r4
                        + c1 * c3 * (dt**2) * r2 * r4
                        + c1 * (dt**3) * r1
                        + c2 * (c3**2) * dt * r1 * r3 * r4
                        + c2 * c3 * (dt**2) * r1 * r3
                        + c2 * c3 * (dt**2) * r1 * r4
                        + c2 * c3 * (dt**2) * r3 * r4
                        + (c3**2) * (dt**2) * r1 * r4
                        + c3 * (dt**3) * r1
                        + c3 * (dt**3) * r4
                    )
                )

            def calc_vc2(
                c1,
                c2,
                c3,
                dt,
                r1,
                r2,
                r3,
                r4,
                v_c1_0,
                v_c2_0,
                v_c3_0,
                v_control,
                v_out,
                v_ref,
            ):
                return (
                    c1 * c2 * (c3**2) * r1 * r2 * r3 * r4 * v_c2_0
                    + c1 * c2 * c3 * dt * r1 * r2 * r3 * v_c2_0
                    + c1 * c2 * c3 * dt * r1 * r2 * r4 * v_c2_0
                    + c1 * c2 * c3 * dt * r1 * r3 * r4 * v_c2_0
                    + c1 * c2 * c3 * dt * r2 * r3 * r4 * v_c2_0
                    + c1 * c2 * (dt**2) * r1 * r3 * v_c2_0
                    - c1 * (c3**2) * dt * r1 * r2 * r4 * v_c3_0
                    + c1 * (c3**2) * dt * r1 * r2 * r4 * v_control
                    - c1 * (c3**2) * dt * r1 * r2 * r4 * v_out
                    - c1 * c3 * (dt**2) * r1 * r2 * v_out
                    - c1 * c3 * (dt**2) * r1 * r4 * v_c1_0
                    + c1 * c3 * (dt**2) * r1 * r4 * v_control
                    - c1 * c3 * (dt**2) * r1 * r4 * v_out
                    - c1 * (dt**3) * r1 * v_out
                    + c1 * (dt**3) * r1 * v_ref
                    + c2 * (c3**2) * dt * r1 * r3 * r4 * v_c2_0
                    + c2 * c3 * (dt**2) * r1 * r3 * v_c2_0
                    + c2 * c3 * (dt**2) * r1 * r4 * v_c2_0
                    + c2 * c3 * (dt**2) * r3 * r4 * v_c2_0
                    - (c3**2) * (dt**2) * r1 * r4 * v_c3_0
                    + (c3**2) * (dt**2) * r1 * r4 * v_control
                    - (c3**2) * (dt**2) * r1 * r4 * v_out
                    - c3 * (dt**3) * r1 * v_out
                ) / (
                    c1 * c2 * (c3**2) * r1 * r2 * r3 * r4
                    + c1 * c2 * c3 * dt * r1 * r2 * r3
                    + c1 * c2 * c3 * dt * r1 * r2 * r4
                    + c1 * c2 * c3 * dt * r1 * r3 * r4
                    + c1 * c2 * c3 * dt * r2 * r3 * r4
                    + c1 * c2 * (dt**2) * r1 * r3
                    + c1 * (c3**2) * dt * r1 * r2 * r4
                    + c1 * c3 * (dt**2) * r1 * r2
                    + c1 * c3 * (dt**2) * r1 * r4
                    + c1 * c3 * (dt**2) * r2 * r4
                    + c1 * (dt**3) * r1
                    + c2 * (c3**2) * dt * r1 * r3 * r4
                    + c2 * c3 * (dt**2) * r1 * r3
                    + c2 * c3 * (dt**2) * r1 * r4
                    + c2 * c3 * (dt**2) * r3 * r4
                    + (c3**2) * (dt**2) * r1 * r4
                    + c3 * (dt**3) * r1
                    + c3 * (dt**3) * r4
                )

            def calc_vc3(
                c1,
                c2,
                c3,
                dt,
                r1,
                r2,
                r3,
                r4,
                v_c1_0,
                v_c2_0,
                v_c3_0,
                v_control,
                v_out,
                v_ref,
            ):
                return (
                    c1 * c2 * pow(c3, 2) * r1 * r2 * r3 * r4 * v_c3_0
                    + c1 * c2 * c3 * dt * r1 * r2 * r3 * v_control
                    - c1 * c2 * c3 * dt * r1 * r2 * r4 * v_c2_0
                    + c1 * c2 * c3 * dt * r1 * r2 * r4 * v_control
                    - c1 * c2 * c3 * dt * r1 * r2 * r4 * v_out
                    + c1 * c2 * c3 * dt * r1 * r3 * r4 * v_c1_0
                    + c1 * c2 * c3 * dt * r2 * r3 * r4 * v_control
                    - c1 * c2 * c3 * dt * r2 * r3 * r4 * v_out
                    + c1 * c2 * pow(dt, 2) * r1 * r3 * v_control
                    - c1 * c2 * pow(dt, 2) * r1 * r3 * v_ref
                    + c1 * pow(c3, 2) * dt * r1 * r2 * r4 * v_c3_0
                    + c1 * c3 * pow(dt, 2) * r1 * r2 * v_control
                    + c1 * c3 * pow(dt, 2) * r1 * r4 * v_c1_0
                    + c1 * c3 * pow(dt, 2) * r2 * r4 * v_control
                    - c1 * c3 * pow(dt, 2) * r2 * r4 * v_out
                    + c1 * pow(dt, 3) * r1 * v_control
                    - c1 * pow(dt, 3) * r1 * v_ref
                    + c2 * pow(c3, 2) * dt * r1 * r3 * r4 * v_c3_0
                    + c2 * c3 * pow(dt, 2) * r1 * r3 * v_control
                    - c2 * c3 * pow(dt, 2) * r1 * r4 * v_c2_0
                    + c2 * c3 * pow(dt, 2) * r1 * r4 * v_control
                    - c2 * c3 * pow(dt, 2) * r1 * r4 * v_out
                    + c2 * c3 * pow(dt, 2) * r3 * r4 * v_control
                    - c2 * c3 * pow(dt, 2) * r3 * r4 * v_out
                    + pow(c3, 2) * pow(dt, 2) * r1 * r4 * v_c3_0
                    + c3 * pow(dt, 3) * r1 * v_control
                    + c3 * pow(dt, 3) * r4 * v_control
                    - c3 * pow(dt, 3) * r4 * v_out
                ) / (
                    c1 * c2 * pow(c3, 2) * r1 * r2 * r3 * r4
                    + c1 * c2 * c3 * dt * r1 * r2 * r3
                    + c1 * c2 * c3 * dt * r1 * r2 * r4
                    + c1 * c2 * c3 * dt * r1 * r3 * r4
                    + c1 * c2 * c3 * dt * r2 * r3 * r4
                    + c1 * c2 * pow(dt, 2) * r1 * r3
                    + c1 * pow(c3, 2) * dt * r1 * r2 * r4
                    + c1 * c3 * pow(dt, 2) * r1 * r2
                    + c1 * c3 * pow(dt, 2) * r1 * r4
                    + c1 * c3 * pow(dt, 2) * r2 * r4
                    + c1 * pow(dt, 3) * r1
                    + c2 * pow(c3, 2) * dt * r1 * r3 * r4
                    + c2 * c3 * pow(dt, 2) * r1 * r3
                    + c2 * c3 * pow(dt, 2) * r1 * r4
                    + c2 * c3 * pow(dt, 2) * r3 * r4
                    + pow(c3, 2) * pow(dt, 2) * r1 * r4
                    + c3 * pow(dt, 3) * r1
                    + c3 * pow(dt, 3) * r4
                )

            v_c1 = calc_vc1(
                c1,
                c2,
                c3,
                dt,
                r1,
                r2,
                r3,
                r4,
                v_c1_0,
                v_c2_0,
                v_c3_0,
                v_control,
                v_out,
                v_ref,
            )
            v_c2 = calc_vc2(
                c1,
                c2,
                c3,
                dt,
                r1,
                r2,
                r3,
                r4,
                v_c1_0,
                v_c2_0,
                v_c3_0,
                v_control,
                v_out,
                v_ref,
            )
            v_c3 = calc_vc3(
                c1,
                c2,
                c3,
                dt,
                r1,
                r2,
                r3,
                r4,
                v_c1_0,
                v_c2_0,
                v_c3_0,
                v_control,
                v_out,
                v_ref,
            )
            v_in_neg = v_control - v_c3
            i_c3 = c3 * (v_c3 - v_c3_0) / dt
            i_c2 = c2 * (v_c2 - v_c2_0) / dt
            i_c1 = c1 * (v_c1 - v_c1_0) / dt

        # recalculate v_c1 due to v_control saturation
        # v_c1 = v_control - v_ref - i_c1 * r2
        debug_state_variables = False
        if debug_state_variables:
            if v_control == vsupply_neg or v_control == vsupply_pos:
                print("saturation occurred")
            print(f"v_in_neg: {v_in_neg}")
            print(f"v_out: {v_out}")
            print(f"v_ref: {v_ref}")
            print(f"v_control: {v_control}")
            print(f"v_c1: {v_c1}")
            print(f"v_c2: {v_c2}")
            print(f"v_c3: {v_c3}")
            print(f"i_c3: {i_c3}")
            print(f"i_c1: {i_c1}")
            print()

        # assign state variables
        self.v_c1 = v_c1
        self.v_c2 = v_c2
        self.v_c3 = v_c3
        self.v_control = v_control

        return v_control
