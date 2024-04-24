import matplotlib.pyplot as plt
import numpy as np
import buck_converter as bc
import voltage_mode_controller as vmc

def app():
    fs = 2e6
    Ts = 1 / fs
    simulation_length_seconds = 0.05

    simulation_sample_length = int(simulation_length_seconds / Ts)

    input_voltage = []
    input_voltage.extend([10] * simulation_sample_length)

    vref = []
    vref.extend([5] * int(simulation_sample_length))

    rload = [10e6] * simulation_sample_length
    # rload = np.logspace(-3, 7, num=simulation_sample_length)
    # rload = rload[::-1]

    type1_compensator = vmc.analog_type_1_with_dc_gain_controller(
        rg=10e3, rf=1e3, cf=470e-9, vsupply_neg=0, vsupply_pos=1
    )

    buck = bc.buck_converter(
        L=200e-6,
        Lesr=0.1,
        C=220e-6,
        Cesr=0.2,
        Rsource=0,
        Rload=rload,
        Vdiode=0.6,
        Rdiode=0,
        controller=type1_compensator,
        synchronous=True,
        # output_current_limit=5,
        # output_voltage_limit=50,
    )
    buck.simulate(
        fs,
        simulation_length_seconds,
        input_voltage,
        Vref=vref,
        # pwm_duty_cycle=1,
        pwm_frequency=10e3,
        pwm_Nskip=0,
    )

    plt.show()


if __name__ == "__main__":
    app()
