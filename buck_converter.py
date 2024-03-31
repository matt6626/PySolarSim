class buck_converter:
    def __init__(
        self,
        L=200e-3,
        Lesr=0.1,
        C=220e-6,
        Cesr=0.2,
        Rsource=0,
        Rload=1,
        Vdiode=0.6,
        Rdiode=0,
    ) -> None:
        self.L = L
        self.Lesr = Lesr
        self.C = C
        self.Cesr = Cesr
        self.Rsource = Rsource
        self.Rload = Rload
        self.Vdiode = Vdiode
        self.Rdiode = Rdiode

    def on_state(self, t0, PA, Vin, Rs, il0, Rl, L, vc0, ESR, C, R):
        K1 = C + ESR * C / R
        KA = -ESR * C / (K1 * L) - (Rl + Rs) / L
        KB = ESR * C / (R * K1 * L) - 1 / L
        vc = vc0 * (1 - PA / (R * K1)) + il0 * (PA / K1)
        il = Vin * PA / L + il0 * (1 + KA * PA) + vc0 * (KB * PA)
        ic = C * (vc - vc0) / PA
        vo = ESR * ic + vc
        t = t0 + PA
        return t, il, ic, vc, vo

    def off_state(self, t0, PA, Vd, Rd, il0, Rl, L, vc0, ESR, C, R):
        K1 = C + ESR * C / R
        KA = -ESR * C / (K1 * L) - (Rl + Rd) / L
        KB = ESR * C / (R * K1 * L) - 1 / L
        vc = vc0 * (1 - PA / (R * K1)) + il0 * (PA / K1)
        il = -Vd * PA / L + il0 * (1 + KA * PA) + vc0 * (KB * PA)
        ic = C * (vc - vc0) / PA
        vo = ESR * ic + vc
        t = t0 + PA
        return t, il, ic, vc, vo

    def PWM_generator(self, P, DC, Ts, TAM):
        import numpy as np

        PA = Ts
        PWM = np.zeros(TAM)
        t = np.zeros(TAM)
        time = 0
        t0 = 0
        i = 0

        while i < TAM:
            if t0 < P:
                if t0 < DC * P:
                    PWM[i] = 1
                    t0 += PA
                else:
                    PWM[i] = 0
                    t0 += PA

            else:
                t0 = 0

            time += PA
            t[i] = time
            i += 1

        return t, PWM

    def saw_tooth_generator(self, Amp, P, tf, TAM):
        import numpy as np

        PA = tf / TAM
        signal = np.zeros(TAM)
        t = np.zeros(TAM)
        time = 0
        t0 = 0
        i = 0

        while t0 < P:
            t0 += PA
            i += 1
        VS = Amp / i
        i = 0
        t0 = 0
        value = 0
        time = 0
        while i < TAM:
            if t0 >= P:
                if t0 >= P:
                    t0 = 0
                    value = 0
                    signal[i] = value
                else:
                    t0 += PA
                    value += VS
                    signal[i] = value
            time += PA
            t[i] = time
            i += 1
        return t, signal

    def controller(self, t0, PA, Vin, Vref, vc0, Rg, Rf, Cf):
        ic = Vin - Vref / Rg
        vc = ic / Cf * PA + vc0
        vo = vc + Rf * ic + Vref
        t = t0 + PA
        return t, ic, vc, vo

    def comparator(self, input, reference):
        output = False
        if input > reference:
            output = True
        else:
            output = False
        return output

    def simulate_fixed_duty(self, fs, sim_length_seconds, Vin, pwm_duty_cycle):
        Ts = 1 / fs
        PA = Ts

        simulation_sample_length = int(sim_length_seconds / Ts)

        print(simulation_sample_length)
        print(Ts * simulation_sample_length)

        # Load stored parameters to local variables
        L = self.L
        Lesr = self.Lesr
        C = self.C
        Cesr = self.Cesr
        Rsource = self.Rsource
        Rload = self.Rload
        Vdiode = self.Vdiode
        Rdiode = self.Rdiode

        # Fixed PWM generation
        pwm_frequency = 10e3
        pwm_period = 1 / pwm_frequency
        pwm_time, pwm_value = self.PWM_generator(
            pwm_period, pwm_duty_cycle, Ts, simulation_sample_length
        )

        # Initial conditions
        il0 = 0
        vc0 = 0

        # Simulation Variables
        il = [il0] * simulation_sample_length
        ic = [0] * simulation_sample_length
        vc = [vc0] * simulation_sample_length
        vo = [0] * simulation_sample_length
        simulation_time = [0] * simulation_sample_length
        # Simulation loop
        simulation_sample_time = 0
        while simulation_sample_time + 1 < simulation_sample_length:
            simulation_sample_time += 1
            simulation_time[simulation_sample_time] = (
                simulation_time[simulation_sample_time - 1] + Ts
            )

            if pwm_value[simulation_sample_time]:
                # On state
                (
                    discard,
                    il[simulation_sample_time],
                    ic[simulation_sample_time],
                    vc[simulation_sample_time],
                    vo[simulation_sample_time],
                ) = self.on_state(
                    simulation_time[simulation_sample_time],
                    PA,
                    Vin,
                    Rsource,
                    il[simulation_sample_time - 1],
                    Lesr,
                    L,
                    vc[simulation_sample_time - 1],
                    Cesr,
                    C,
                    Rload,
                )
            else:
                # Off state
                (
                    discard,
                    il[simulation_sample_time],
                    ic[simulation_sample_time],
                    vc[simulation_sample_time],
                    vo[simulation_sample_time],
                ) = self.off_state(
                    simulation_time[simulation_sample_time],
                    PA,
                    Vdiode,
                    Rdiode,
                    il[simulation_sample_time - 1],
                    Lesr,
                    L,
                    vc[simulation_sample_time - 1],
                    Cesr,
                    C,
                    Rload,
                )

        # Plot simulation
        import matplotlib.pyplot as plt
        import mplcursors

        # Create the subplots
        fig, axs = plt.subplots(5, figsize=(10, 30), sharex=True)

        # Plot Induction Current
        axs[0].plot(simulation_time, il, label="iL")
        axs[0].set_xlabel("time (s)")
        axs[0].set_ylabel("iL (A)")
        axs[0].legend()
        axs[0].grid(True)

        # Plot Capacitor Current
        axs[1].plot(simulation_time, ic, label="ic")
        axs[1].set_xlabel("time (s)")
        axs[1].set_ylabel("ic (A)")
        axs[1].legend()
        axs[1].grid(True)

        # Plot Capacitor Voltage
        axs[2].plot(simulation_time, vc, label="vc")
        axs[2].set_xlabel("time (s)")
        axs[2].set_ylabel("vc")
        axs[2].legend()
        axs[2].grid(True)

        # Plot Output Voltage
        axs[3].plot(simulation_time, vo, label="vo")
        axs[3].set_xlabel("time (s)")
        axs[3].set_ylabel("vo")
        axs[3].legend()
        axs[3].grid(True)

        # Plot PWM
        axs[4].plot(simulation_time, pwm_value, label="pwm")
        axs[4].set_xlabel("time (s)")
        axs[4].set_ylabel("pwm")
        axs[4].legend()
        axs[4].grid(True)

        # Add a linked cursor
        cursor = mplcursors.cursor(axs, hover=True)

        # Define the annotation for the cursor
        @cursor.connect("add")
        def on_add(sel):
            x, y = sel.target
            sel.annotation.set_text(f"x: {x:.2f}\ny: {y:.2f}")

        # Add a title to the figure
        fig.suptitle("Buck Converter Simulation")

        # Display the plot
        plt.show(block=False)


fs = 1e6
buck = buck_converter()
buck.simulate_fixed_duty(fs, 1, 10, 0.5)
import matplotlib.pyplot as plt

plt.show()
