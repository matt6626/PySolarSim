import matplotlib.pyplot as plt
import numpy as np
import buck_converter as bc
import voltage_mode_controller as vmc

fs = 2e6
Ts = 1 / fs
simulation_length_seconds = 1

simulation_sample_length = int(simulation_length_seconds / Ts)

input_voltage = []
input_voltage.extend([50] * simulation_sample_length)

vref = []
vref.extend([25] * int(simulation_sample_length))

# rload = [10e6] * simulation_sample_length
rload = np.logspace(-3, 7, num=simulation_sample_length)
rload = rload[::-1]

analog_type1_with_dc_gain_controller_control_func_params = {
    "v_cf": 0,
    "rg": 100e3,
    "rf": 0,
    "cf": 470e-9,
    "vsupply_neg": 0,
    "vsupply_pos": 1,
}

analog_type1_with_dc_gain_controller = vmc.voltage_mode_controller(
    analog_type1_with_dc_gain_controller_control_func_params,
    vmc.analog_type1_with_dc_gain_controller_control_func,
)

buck = bc.buck_converter(
    L=10e-6,
    Lesr=0.1,
    C=1000e-6,
    Cesr=0.1,
    Rsource=0,
    Rload=rload,
    Vdiode=0.6,
    Rdiode=0,
    Rg=100e3,
    Rf=0,
    Cf=470e-9,
    controller=analog_type1_with_dc_gain_controller,
)
buck.simulate(
    fs,
    simulation_length_seconds,
    input_voltage,
    Vref=vref,
    # pwm_duty_cycle=1,
    pwm_frequency=20e3,
    pwm_Nskip=0,
    output_current_limit=5,
    output_voltage_limit=50,
)

plt.show()
