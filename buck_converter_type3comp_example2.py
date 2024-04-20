import matplotlib.pyplot as plt
import numpy as np
import buck_converter as bc
import voltage_mode_controller as vmc

# Example 2
# Based on https://www.youtube.com/watch?v=abelKfvEtgE

fs = 10e6
Ts = 1 / fs
simulation_length_seconds = 0.02

simulation_sample_length = int(simulation_length_seconds / Ts)

input_voltage_amplitude = 20
input_voltage = []
input_voltage.extend([input_voltage_amplitude] * simulation_sample_length)

vref = []
vref.extend([5] * int(simulation_sample_length))
from filters import rc_filter

vref = rc_filter(vref, 1e3, 1e-6, 0, Ts)

# rload = [10e6] * simulation_sample_length
rload = np.logspace(-3, 7, num=simulation_sample_length)
rload = rload[::-1]

# this type3 compensator was tuned using this calculator:
# https://view.officeapps.live.com/op/view.aspx?src=https%3A%2F%2Fassets.maxlinear.com%2Fweb%2Fdocuments%2Fsipex%2Fapplicationnotes%2Fanp-16_typeiiicalculator_101206.xls&wdOrigin=BROWSELINK
# fc = 2 kHz
# fsw = 20 kHz
vref_gain = 1
analog_type3_controller = vmc.analog_type3_compensator_controller(
    r1=8.2e3,
    r2=22e3,
    r3=4.7e3,
    r4=10e12,
    c1=15e-9,
    c2=12e-9,
    c3=100e-12,
    vsupply_neg=0,
    vsupply_pos=5,
    open_loop_gain=10e6,
)

buck = bc.buck_converter(
    L=56e-6,
    Lesr=0.01,
    C=500e-6,
    Cesr=0.01,
    Rsource=0,
    Rload=rload,
    Vdiode=0.6,
    Rdiode=0,
    output_current_limit=5,
    inductor_current_limit=5,
    synchronous=True,
    controller=analog_type3_controller,
)
buck.analyse(RLOAD=10e6, bode_plant=True)
buck.simulate(
    fs,
    simulation_length_seconds,
    input_voltage,
    Vref=vref,
    vref_gain=vref_gain,
    # pwm_duty_cycle=1,
    pwm_frequency=100e3,
    pwm_Nskip=0,
)

plt.show()
