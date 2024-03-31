import matplotlib.pyplot as plt
import numpy as np
import buck_converter as bc

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

buck = bc.buck_converter()
buck.simulate(fs, simulation_length_seconds, input_voltage, Vref=vref)


plt.show()
