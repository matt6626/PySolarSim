import matplotlib.pyplot as plt
import numpy as np
import buck_converter as bc
import voltage_mode_controller as vmc

fs = 2e6
Ts = 1 / fs
simulation_length_seconds = 0.01

simulation_sample_length = int(simulation_length_seconds / Ts)

input_voltage_amplitude = 50
input_voltage = []
input_voltage.extend([input_voltage_amplitude] * simulation_sample_length)

vref = []
vref.extend([25] * int(simulation_sample_length))

# rload = [10e6] * simulation_sample_length
rload = np.logspace(-3, 7, num=simulation_sample_length)
rload = rload[::-1]

# this type3 compensator was tuned using this calculator:
# https://view.officeapps.live.com/op/view.aspx?src=https%3A%2F%2Fassets.maxlinear.com%2Fweb%2Fdocuments%2Fsipex%2Fapplicationnotes%2Fanp-16_typeiiicalculator_101206.xls&wdOrigin=BROWSELINK
# fc = 2 kHz
# fsw = 20 kHz
analog_type3_controller_control_func_params = {
    "v_c1": 0,
    "v_c2": 0,
    "v_c3": 0,
    "v_control": 0,
    "r1": 100e3,
    "r2": 100e3,
    "r3": 100e3,
    "r4": 2.04e3,
    "c1": 1e-9,
    "c2": 10e-9,
    "c3": 100e-9,
    "vsupply_neg": 0,
    "vsupply_pos": 1,
}

analog_type3_controller = vmc.voltage_mode_controller(
    analog_type3_controller_control_func_params,
    vmc.analog_type3_compensator_control_func,
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
    controller=analog_type3_controller,
)
buck.simulate(
    fs,
    simulation_length_seconds,
    input_voltage,
    Vref=vref,
    vref_gain=(1 / input_voltage_amplitude),
    # pwm_duty_cycle=1,
    pwm_frequency=20e3,
    pwm_Nskip=2,
    output_current_limit=5,
    output_voltage_limit=50,
)

plt.show()
