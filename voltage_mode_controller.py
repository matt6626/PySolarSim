class voltage_mode_controller:
    def __init__(
        self,
        control_func_params={},
        control_func=(lambda x, y, dt: y),
    ) -> None:
        self.simulation_not_started = True
        self.control_func = control_func
        self.control_func_params = control_func_params

    def simulate(self, vref, vout, dt, plot=False):
        if self.simulation_not_started:
            # TODO: try not to run this before control_func is called so that optional parameters don't break things
            # default dict makes more sense!!!! then above suggestion
            if plot:
                self.plot(vref, vout, dt, self.control_func_params, init=True)
            self.simulation_not_started = False
        vcontrol = self.control_func(vref, vout, dt, self.control_func_params)
        if plot:
            self.plot(vref, vout, dt, self.control_func_params)
        return vcontrol

    def plot(self, vref, vout, dt, control_func_params: dict = {}, init=False):
        import matplotlib.pyplot as plt
        import numpy as np
        import matplotlib.animation as animation

        # TODO: plotting in realtime is unbelievably slow in current implementation

        if init:

            # Combine the default variables with the control function variables
            self.vars = {
                "vref": [vref],
                "vout": [vout],
                "dt": [dt],
            }
            for key, value in control_func_params.items():
                self.vars[key] = [value]

            # Perform initial plot
            self.t = [0]

            # Generate Plot
            plt.ion()
            # plt.autoscale(enable=True, axis="both", tight=True)
            self.fig = plt.figure()
            self.nplots = len(self.vars)
            for i, (key, value) in enumerate(self.vars.items()):
                value = [value, self.fig.add_subplot(self.nplots, 1, i + 1)]
                value[1].plot(self.t[0], value[0])
                value[1].set(xlabel="time (s)", ylabel=f"{key}")
                value[1].relim()
                value[1].autoscale_view()
                self.vars[key] = value  # update the actual dict
        else:
            # Update
            self.t.append(self.t[-1] + dt)

            # Append values to memory
            # print(self.vars["vref"])
            self.vars["vref"][0].append(vref)
            self.vars["vout"][0].append(vout)
            self.vars["dt"][0].append(dt)
            for key, value in control_func_params.items():
                self.vars[key][0].append(value)

            for key, value in self.vars.items():
                value[1].relim()
                value[1].autoscale_view()
                value[1].get_lines()[0].set_xdata(self.t)
                value[1].get_lines()[0].set_ydata(value[0])

            self.fig.canvas.draw()
            self.fig.canvas.flush_events()


def analog_type1_with_dc_gain_controller_control_func(vref, vout, dt, param={}):
    # circuit
    #                          ___      ||
    #                      ---|___|-----||----
    #                      |            ||   |
    #                      |   Rf       Cf   |
    #               Rg     |                 |
    #               ___    |       |\|       |
    #    Vout o----|___|---o-------|-\       |
    #                              |  >------o----o Vcontrol
    #    Vref o--------------------|+/
    #                              |/|

    # required controller imports
    import numpy as np

    # memory variables
    if param.get("v_cf") is None:
        raise Exception("vcf is a required memory variable")

    # controller parameter defaults
    if param.get("rg") is None:
        param["rg"] = 1e3
    if param.get("rf") is None:
        param["rf"] = 0
    if param.get("cf") is None:
        param["cf"] = 1e-9
    if param.get("vsupply_neg") is None:
        param["vsupply_neg"] = -np.Inf
    if param.get("vsupply_pos") is None:
        param["vsupply_pos"] = np.Inf

    i_cf = (vref - vout) / param["rg"]
    v_cf = ((i_cf / param["cf"]) * dt) + param["v_cf"]
    v_control = np.clip(
        v_cf + (param["rf"] * i_cf) + vref, param["vsupply_neg"], param["vsupply_pos"]
    )
    # TODO: revisit the saturation calculations as vin- is not currently recalculated so results will be wrong
    # recalculate feedback capacitor voltage after controller output saturates
    v_cf = v_control - (i_cf * param["rf"]) - vref

    # assign new values to memory
    param["v_cf"] = v_cf

    return v_control


def analog_type3_compensator_control_func(vref, vout, dt, param={}):
    # circuit
    # https://www.ti.com/lit/an/slva662/slva662.pdf

    # required controller imports
    import numpy as np

    # memory variables
    if param.get("v_c1") is None:
        raise Exception("v_c1 is a required memory variable")
    if param.get("v_c2") is None:
        raise Exception("v_c2 is a required memory variable")
    if param.get("v_c3") is None:
        raise Exception("v_c3 is a required memory variable")
    if param.get("v_control") is None:
        param["v_control"] = 0

    # controller parameter defaults
    if param.get("r1") is None:
        param["r1"] = 1e3
    if param.get("r2") is None:
        param["r2"] = 1e3
    if param.get("r3") is None:
        param["r3"] = 1e3
    if param.get("r4") is None:
        param["r4"] = 1e3
    if param.get("c1") is None:
        param["c1"] = 1e-9
    if param.get("c2") is None:
        param["c2"] = 1e-9
    if param.get("c3") is None:
        param["c3"] = 1e-9

    if param.get("vsupply_neg") is None:
        param["vsupply_neg"] = -np.Inf
    if param.get("vsupply_pos") is None:
        param["vsupply_pos"] = np.Inf

    if param.get("open_loop_gain") is None:
        param["open_loop_gain"] = 10e6

    if vref < param["vsupply_neg"]:
        raise Exception(
            f"vref violates saturation limits (vref={vref}) < vsupply_neg={param['vsupply_neg']}"
        )
    if vref > param["vsupply_pos"]:
        raise Exception(
            f"vref violates saturation limits (vref={vref}) > vsupply_pos={param['vsupply_pos']}"
        )

    # Zi
    i_r1 = (vref - vout) / param["r1"]
    # v_c2 eqn derived from:
    # eqn1: Ic = C (Vc - Vc0) / dt
    # eqn2: Vc = Vref - Vout - Ic * R3
    # eqn1 -> eqn2: Vc = Vref - Vout - C * R3 * (Vc - Vc0) / dt
    v_c2 = (vref - vout + (param["r3"] * param["c2"] * param["v_c2"] / dt)) / (
        1 + (param["r3"] * param["c2"] / dt)
    )
    i_c2 = param["c2"] * (v_c2 - param["v_c2"]) / dt
    i_zi = i_r1 + i_c2

    # Zf
    # equations:
    # eqn1: v_c3 = v_control - vref
    # eqn2: i_c3 = c3 * (v_c3 - v_c3_0) / dt
    # eqn1 => eqn2 (eqn3): i_c3 = c3 * (v_control - vref - v_c3_0) / dt
    # eqn4: i_c3 + i_c1 = i_zi + vref / r4
    # eqn4 => eqn3 (eqn5): i_zi + vref / r4 - i_c1 = c3 * (v_control - vref - v_c3_0) / dt
    # eqn6: i_c1 = c1 * (v_c1 - v_c1_0) / dt
    # eqn6 => eqn5 (eqn7): i_zi + vref / r4 - c1 * (v_c1 - v_c1_0) / dt = c3 * (v_control - vref - v_c3_0) / dt
    # rearrange eqn7 (eqn8***): i_zi * dt + vref / r4 * dt - c1 * (v_c1 - v_c1_0) = c3 * (v_control - vref - v_c3_0)
    # eqn9: v_c1 = v_control - vref - i_c1 * r2
    # eqn6 => eqn9 (eqn10***): v_c1 = v_control - vref - r2 * c1 * (v_c1 - v_c1_0) / dt
    # use eqn8, eqn10 to solve for v_c1 and v_control:
    # rearrange eqn8 (eqn11): v_control = (i_zi * dt + vref / r4 * dt - c1 * (v_c1 - v_c1_0)) / c3 + vref + v_c3_0
    # eqn11 => eqn10 (eqn12): v_c1 + c1 * v_c1 / c3 + r2 * c1 * v_c1 / dt = i_zi * dt / c3 + vref * dt / (r4 * c3) + c1 * v_c1_0 / c3 + v_c3_0 + r2 * c1 * v_c1_0 / dt
    # simplify eqn12 (eqn13): v_c1 = (i_zi * dt / c3 + vref * dt / (r4 * c3) + c1 * v_c1_0 / c3 + v_c3_0 + r2 * c1 * v_c1_0 / dt) / (1 + c1 / c3 + r2 * c1 / dt)
    # then arrange eqn10 to solve for v_control (eqn14): v_control = v_c1 + vref + r2 * c1 * (v_c1 - v_c1_0) / dt
    v_c1 = (
        i_zi * dt / param["c3"]
        + vref * dt / (param["r4"] * param["c3"])
        + param["c1"] * param["v_c1"] / param["c3"]
        + param["v_c3"]
        + param["r2"] * param["c1"] * param["v_c1"] / dt
    ) / (1 + param["c1"] / param["c3"] + param["r2"] * param["c1"] / dt)
    v_control = np.clip(
        v_c1 + vref + param["r2"] * param["c1"] * (v_c1 - param["v_c1"]) / dt,
        param["vsupply_neg"],
        param["vsupply_pos"],
    )

    v_c3 = v_control - vref
    i_c3 = param["c3"] * (v_c3 - param["v_c3"]) / dt
    i_c1 = i_zi + vref / param["r4"] - i_c3
    v_in_neg = v_control - v_c3

    # opamp saturation
    # if saturation occurred, calculate the true voltage at the inverting terminal, and recalculate circuit voltages/currents as result
    # should be able to always perform this calculation
    if v_control == param["vsupply_neg"] or v_control == param["vsupply_pos"]:
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
        # v_c1 = [i_zi * dt / c3 + vref * dt / (r4 * c3) + c1 * v_c1_0 / c3 + v_c3_0 + r2 * c1 * v_c1_0 / dt] / [1 + c1 / c3 + r2 * c1 / dt]
        # v_c3 = v_control - v_in_neg
        # i_c3 = c3 * (v_c3 - v_c3_0) / dt
        # i_c1 = c1 * (v_c1 - v_c1_0) / dt
        # i_zf = i_c1 + i_c3

        # substitute (*) => Zf equations:
        # v_c1 = [(i_r1 + i_c2) * dt / c3 + vref * dt / (r4 * c3) + c1 * v_c1_0 / c3 + v_c3_0 + r2 * c1 * v_c1_0 / dt] / [1 + c1 / c3 + r2 * c1 / dt]
        # v_c3 = v_control - ( i_c1 + i_c3 - i_r1 - i_c2)* r4

        # list all our equations:
        # (1):
        # v_c1 = [(i_r1 + i_c2) * dt / c3 + vref * dt / (r4 * c3) + c1 * v_c1_0 / c3 + v_c3_0 + r2 * c1 * v_c1_0 / dt] / [1 + c1 / c3 + r2 * c1 / dt]

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
        # v_c1 = [(([(c1 * (v_c1 - v_c1_0) / dt + c3 * (v_c3 - v_c3_0) / dt - c2 * (v_c2 - v_c2_0) / dt)* r4 - v_out] / [r1 + r4]) + c2 * (v_c2 - v_c2_0) / dt) * dt / c3 + vref * dt / (r4 * c3) + c1 * v_c1_0 / c3 + v_c3_0 + r2 * c1 * v_c1_0 / dt] / [1 + c1 / c3 + r2 * c1 / dt]

        # (9):
        # v_c2 = [c1 * (v_c1 - v_c1_0) / dt + c3 * (v_c3 - v_c3_0) / dt - i_r1 - c2 * (v_c2 - v_c2_0) / dt)* r4 - v_out + (r3 * c2 * v_c2_0 / dt)] / [1 + (r3 * c2 / dt)]

        # (10):
        # v_c3 = v_control - (c1 * (v_c1 - v_c1_0) / dt + c3 * (v_c3 - v_c3_0) / dt - ([(c1 * (v_c1 - v_c1_0) / dt + c3 * (v_c3 - v_c3_0) / dt - c2 * (v_c2 - v_c2_0) / dt)* r4 - v_out] / [r1 + r4]) - c2 * (v_c2 - v_c2_0) / dt)* r4

        # if you do all the maths, then use sympy to solve the simulatenous equations
        def calc_vc1(
            c1, c2, c3, dt, r1, r2, r3, r4, v_c1_0, v_c2_0, v_c3_0, vcontrol, vout, vref
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
                + c2 * c3 * (dt**2) * r1 * r3 * r4 * vref
                - c2 * c3 * (dt**2) * r1 * (r4**2) * v_c2_0
                + c2 * c3 * (dt**2) * r1 * (r4**2) * vcontrol
                - c2 * c3 * (dt**2) * r1 * (r4**2) * vout
                + c2 * c3 * (dt**2) * r3 * (r4**2) * vcontrol
                - c2 * c3 * (dt**2) * r3 * (r4**2) * vout
                + c2 * (dt**3) * r1 * r3 * vref
                - c2 * (dt**3) * r1 * r4 * v_c2_0
                - c2 * (dt**3) * r1 * r4 * vout
                + c2 * (dt**3) * r1 * r4 * vref
                - c2 * (dt**3) * r3 * r4 * vout
                + c2 * (dt**3) * r3 * r4 * vref
                + (c3**2) * (dt**2) * r1 * (r4**2) * v_c3_0
                + c3 * (dt**3) * r1 * r4 * v_c3_0
                + c3 * (dt**3) * r1 * r4 * vref
                + c3 * (dt**3) * (r4**2) * vcontrol
                - c3 * (dt**3) * (r4**2) * vout
                + (dt**4) * r1 * vref
                - (dt**4) * r4 * vout
                + (dt**4) * r4 * vref
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
            c1, c2, c3, dt, r1, r2, r3, r4, v_c1_0, v_c2_0, v_c3_0, vcontrol, vout, vref
        ):
            return (
                c1 * c2 * (c3**2) * r1 * r2 * r3 * r4 * v_c2_0
                + c1 * c2 * c3 * dt * r1 * r2 * r3 * v_c2_0
                + c1 * c2 * c3 * dt * r1 * r2 * r4 * v_c2_0
                + c1 * c2 * c3 * dt * r1 * r3 * r4 * v_c2_0
                + c1 * c2 * c3 * dt * r2 * r3 * r4 * v_c2_0
                + c1 * c2 * (dt**2) * r1 * r3 * v_c2_0
                - c1 * (c3**2) * dt * r1 * r2 * r4 * v_c3_0
                + c1 * (c3**2) * dt * r1 * r2 * r4 * vcontrol
                - c1 * (c3**2) * dt * r1 * r2 * r4 * vout
                - c1 * c3 * (dt**2) * r1 * r2 * vout
                - c1 * c3 * (dt**2) * r1 * r4 * v_c1_0
                + c1 * c3 * (dt**2) * r1 * r4 * vcontrol
                - c1 * c3 * (dt**2) * r1 * r4 * vout
                - c1 * (dt**3) * r1 * vout
                + c1 * (dt**3) * r1 * vref
                + c2 * (c3**2) * dt * r1 * r3 * r4 * v_c2_0
                + c2 * c3 * (dt**2) * r1 * r3 * v_c2_0
                + c2 * c3 * (dt**2) * r1 * r4 * v_c2_0
                + c2 * c3 * (dt**2) * r3 * r4 * v_c2_0
                - (c3**2) * (dt**2) * r1 * r4 * v_c3_0
                + (c3**2) * (dt**2) * r1 * r4 * vcontrol
                - (c3**2) * (dt**2) * r1 * r4 * vout
                - c3 * (dt**3) * r1 * vout
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
            c1, c2, c3, dt, r1, r2, r3, r4, v_c1_0, v_c2_0, v_c3_0, vcontrol, vout, vref
        ):
            return (
                c1 * c2 * pow(c3, 2) * r1 * r2 * r3 * r4 * v_c3_0
                + c1 * c2 * c3 * dt * r1 * r2 * r3 * vcontrol
                - c1 * c2 * c3 * dt * r1 * r2 * r4 * v_c2_0
                + c1 * c2 * c3 * dt * r1 * r2 * r4 * vcontrol
                - c1 * c2 * c3 * dt * r1 * r2 * r4 * vout
                + c1 * c2 * c3 * dt * r1 * r3 * r4 * v_c1_0
                + c1 * c2 * c3 * dt * r2 * r3 * r4 * vcontrol
                - c1 * c2 * c3 * dt * r2 * r3 * r4 * vout
                + c1 * c2 * pow(dt, 2) * r1 * r3 * vcontrol
                - c1 * c2 * pow(dt, 2) * r1 * r3 * vref
                + c1 * pow(c3, 2) * dt * r1 * r2 * r4 * v_c3_0
                + c1 * c3 * pow(dt, 2) * r1 * r2 * vcontrol
                + c1 * c3 * pow(dt, 2) * r1 * r4 * v_c1_0
                + c1 * c3 * pow(dt, 2) * r2 * r4 * vcontrol
                - c1 * c3 * pow(dt, 2) * r2 * r4 * vout
                + c1 * pow(dt, 3) * r1 * vcontrol
                - c1 * pow(dt, 3) * r1 * vref
                + c2 * pow(c3, 2) * dt * r1 * r3 * r4 * v_c3_0
                + c2 * c3 * pow(dt, 2) * r1 * r3 * vcontrol
                - c2 * c3 * pow(dt, 2) * r1 * r4 * v_c2_0
                + c2 * c3 * pow(dt, 2) * r1 * r4 * vcontrol
                - c2 * c3 * pow(dt, 2) * r1 * r4 * vout
                + c2 * c3 * pow(dt, 2) * r3 * r4 * vcontrol
                - c2 * c3 * pow(dt, 2) * r3 * r4 * vout
                + pow(c3, 2) * pow(dt, 2) * r1 * r4 * v_c3_0
                + c3 * pow(dt, 3) * r1 * vcontrol
                + c3 * pow(dt, 3) * r4 * vcontrol
                - c3 * pow(dt, 3) * r4 * vout
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
            param["c1"],
            param["c2"],
            param["c3"],
            dt,
            param["r1"],
            param["r2"],
            param["r3"],
            param["r4"],
            param["v_c1"],
            param["v_c2"],
            param["v_c3"],
            v_control,
            vout,
            vref,
        )
        v_c2 = calc_vc2(
            param["c1"],
            param["c2"],
            param["c3"],
            dt,
            param["r1"],
            param["r2"],
            param["r3"],
            param["r4"],
            param["v_c1"],
            param["v_c2"],
            param["v_c3"],
            v_control,
            vout,
            vref,
        )
        v_c3 = calc_vc3(
            param["c1"],
            param["c2"],
            param["c3"],
            dt,
            param["r1"],
            param["r2"],
            param["r3"],
            param["r4"],
            param["v_c1"],
            param["v_c2"],
            param["v_c3"],
            v_control,
            vout,
            vref,
        )
        v_in_neg = v_control - v_c3
        i_c3 = param["c3"] * (v_c3 - param["v_c3"]) / dt
        i_c2 = param["c2"] * (v_c2 - param["v_c2"]) / dt
        i_c1 = param["c1"] * (v_c1 - param["v_c1"]) / dt

    # recalculate v_c1 due to v_control saturation
    # v_c1 = v_control - vref - i_c1 * param["r2"]
    debug_state_variables = False
    if debug_state_variables:
        if v_control == param["vsupply_neg"] or v_control == param["vsupply_pos"]:
            print("saturation occurred")
        print(f"v_in_neg: {v_in_neg}")
        print(f"vout: {vout}")
        print(f"vref: {vref}")
        print(f"v_control: {v_control}")
        print(f"v_c1: {v_c1}")
        print(f"v_c2: {v_c2}")
        print(f"v_c3: {v_c3}")
        print(f"i_c3: {i_c3}")
        print(f"i_c1: {i_c1}")
        print()

    # assign new values to memory
    param["v_c1"] = v_c1
    param["v_c2"] = v_c2
    param["v_c3"] = v_c3
    param["v_control"] = v_control

    return v_control
