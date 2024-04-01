class voltage_mode_controller:
    def __init__(
        self,
        control_func_params={},
        control_func=(lambda x, y, dt: y),
    ) -> None:
        self.control_func = control_func
        self.control_func_params = control_func_params

    def simulate(self, vref, vout, dt):
        return self.control_func(vref, vout, dt, self.control_func_params)


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
    # recalculate v_c1 due to v_control saturation
    v_c1 = v_control - vref - i_c1 * param["r2"]

    # assign new values to memory
    param["v_c1"] = v_c1
    param["v_c2"] = v_c2
    param["v_c3"] = v_c3

    return v_control
