import matplotlib.pyplot as plt
import numpy as np
import buck_converter as bc
import voltage_mode_controller as vmc

fs = 1e6
Ts = 1 / fs
simulation_length_seconds = 1

simulation_sample_length = int(simulation_length_seconds / Ts)

input_voltage = []
input_voltage.extend([20] * 250000)
input_voltage.extend([10] * 750000)
vref = []
vref.extend([10] * 500000)
vref.extend([5] * 500000)

type1_compensator = vmc.analog_type_1_with_dc_gain_controller(
    rg=10e3, rf=1e3, cf=470e-9
)

buck = bc.buck_converter(
    L=200e-3,
    Lesr=0.1,
    C=220e-6,
    Cesr=0.2,
    Rsource=0,
    Rload=1,
    Vdiode=0.6,
    Rdiode=0,
    controller=type1_compensator,
)
buck.simulate(fs, simulation_length_seconds, input_voltage, Vref=vref)


plt.show()
