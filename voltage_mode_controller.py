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
