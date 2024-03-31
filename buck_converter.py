import numpy as np

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
        Rg=10e3,
        Rf=1e3,
        Cf=470e-9,
    ) -> None:
        self.L = L
        self.Lesr = Lesr
        self.C = C
        self.Cesr = Cesr
        self.Rsource = Rsource
        self.Rload = Rload
        self.Vdiode = Vdiode
        self.Rdiode = Rdiode
        self.Rg = Rg
        self.Rf = Rf
        self.Cf = Cf

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

    def saw_tooth_generator(self, Amp, P, Ts, TAM):
        import numpy as np

        PA = Ts
        signal = np.zeros(TAM)
        t = np.zeros(TAM)
        time = 0
        t0 = 0
        i = 0

        # determine sawtooth step size
        while t0 < P:
            t0 += PA
            i += 1
        VS = Amp / i

        # calculate the sawtooth
        i = 0
        t0 = 0
        value = 0
        time = 0
        while i < TAM:
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

    def controller(
        self,
        t0,
        PA,
        Vin,
        Vref,
        vc0,
        Rg,
        Rf,
        Cf,
        Vsupply_neg=-np.Inf,
        Vsupply_pos=np.Inf,
    ):
        ic = (Vref - Vin) / Rg
        vc = ((ic / Cf) * PA) + vc0
        vo = np.clip(vc + (Rf * ic) + Vref, Vsupply_neg, Vsupply_pos)
        # recalculate feedback capacitor voltage after controller output saturates
        vc = vo - (ic * Rf) - Vref
        t = t0 + PA
        return t, ic, vc, vo

    def comparator(self, input, reference):
        output = 0
        if input > reference:
            output = 1
        else:
            output = 0
        return output

    def simulate(self, fs, sim_length_seconds, Vin, pwm_duty_cycle=None, Vref=0):
        Ts = 1 / fs
        PA = Ts

        simulation_sample_length = int(sim_length_seconds / Ts)

        # Load stored parameters to local variables
        L = self.L
        Lesr = self.Lesr
        C = self.C
        Cesr = self.Cesr
        Rsource = self.Rsource
        Rload = self.Rload
        Vdiode = self.Vdiode
        Rdiode = self.Rdiode

        if pwm_duty_cycle is not None:
            # Fixed PWM generation
            pwm_frequency = 10e3
            pwm_period = 1 / pwm_frequency
            pwm_time, pwm_value = self.PWM_generator(
                pwm_period, pwm_duty_cycle, Ts, simulation_sample_length
            )
        else:
            pwm_value = [0] * simulation_sample_length
            # Fixed sawtooth generation
            pwm_frequency = 10e3
            pwm_period = 1 / pwm_frequency
            sawtooth_time, sawtooth_value = self.saw_tooth_generator(
                1, pwm_period, Ts, simulation_sample_length
            )

        # Initial conditions
        il0 = 0
        vc0 = 0

        # Simulation Variables
        il = [il0] * simulation_sample_length
        ic = [0] * simulation_sample_length
        vc = [vc0] * simulation_sample_length
        vo = [0] * simulation_sample_length
        if len(Vin) == simulation_sample_length:
            vin = Vin
        else:
            vin = [Vin] * simulation_sample_length
        # Controller Variables
        if pwm_duty_cycle is None:
            vcontrol0 = 0
            vcontrol = [vcontrol0] * simulation_sample_length
            if len(Vref) == simulation_sample_length:
                vref = Vref
            else:
                vref = [Vref] * simulation_sample_length
            vcf0 = 0
            vcf = [vcf0] * simulation_sample_length
        # Extra Variables
        verr = [vref[0] - vo[0]] * simulation_sample_length
        # Simulation time
        simulation_time = [0] * simulation_sample_length
        # Simulation loop
        simulation_sample_time = 0
        while simulation_sample_time + 1 < simulation_sample_length:
            simulation_sample_time += 1
            simulation_time[simulation_sample_time] = (
                simulation_time[simulation_sample_time - 1] + Ts
            )

            if pwm_duty_cycle is None:
                # controller simulation
                (
                    discard_time,
                    discard_icf,
                    vcf[simulation_sample_time],
                    vcontrol[simulation_sample_time],
                ) = self.controller(
                    simulation_time[simulation_sample_time],
                    Ts,
                    vo[simulation_sample_time - 1],
                    vref[simulation_sample_time],
                    vcf[simulation_sample_time - 1],
                    self.Rg,
                    self.Rf,
                    self.Cf,
                    0,
                    5,
                )

                pwm_value[simulation_sample_time] = self.comparator(
                    vcontrol[simulation_sample_time],
                    sawtooth_value[simulation_sample_time],
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
                    vin[simulation_sample_time],
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

            verr[simulation_sample_time] = (
                vref[simulation_sample_time] - vo[simulation_sample_time]
            )

        # Plot simulation
        import matplotlib.pyplot as plt
        import mplcursors

        # Create the subplots
        if pwm_duty_cycle is not None:
            fig, axs = plt.subplots(4, figsize=(10, 30), sharex=True)
        else:
            fig, axs = plt.subplots(4, figsize=(10, 30), sharex=True)

        # Plot Input Voltage
        axs[0].plot(simulation_time, vin, label="vin")
        axs[0].set_xlabel("time (s)")
        axs[0].set_ylabel("vin (V)")
        axs[0].legend()
        axs[0].grid(True)

        # Plot Reference Voltage
        if pwm_duty_cycle is None:
            axs[1].plot(simulation_time, vref, label="vref")
        axs[1].set_xlabel("time (s)")
        axs[1].set_ylabel("vref (V)")
        axs[1].legend()
        axs[1].grid(True)

        # Plot Output Voltage
        axs[1].plot(simulation_time, vo, label="vout")
        axs[1].set_xlabel("time (s)")
        axs[1].set_ylabel("vout (V)")
        axs[1].legend()
        axs[1].grid(True)

        # Plot Error Voltage
        axs[2].plot(simulation_time, verr, label="verr")
        axs[2].set_xlabel("time (s)")
        axs[2].set_ylabel("verr (V)")
        axs[2].legend()
        axs[2].grid(True)

        # Plot Control Voltage
        if pwm_duty_cycle is None:
            axs[3].plot(simulation_time, vcontrol, label="vcontrol")
            axs[3].plot(simulation_time, vcf, label="vcf")
        else:
            axs[3].plot(
                simulation_time,
                [pwm_duty_cycle] * simulation_sample_length,
                label="vcontrol",
            )
        axs[3].set_xlabel("time (s)")
        axs[3].set_ylabel("vcontrol (V)")
        axs[3].legend()
        axs[3].grid(True)

        # # Plot Inductor Current
        # axs[1].plot(simulation_time, il, label="iL")
        # axs[1].set_xlabel("time (s)")
        # axs[1].set_ylabel("iL (A)")
        # axs[1].legend()
        # axs[1].grid(True)

        # # Plot Capacitor Current
        # axs[2].plot(simulation_time, ic, label="ic")
        # axs[2].set_xlabel("time (s)")
        # axs[2].set_ylabel("ic (A)")
        # axs[2].legend()
        # axs[2].grid(True)

        # # Plot Capacitor Voltage
        # axs[3].plot(simulation_time, vc, label="vc")
        # axs[3].set_xlabel("time (s)")
        # axs[3].set_ylabel("vc")
        # axs[3].legend()
        # axs[3].grid(True)

        # # Plot Output Voltage
        # axs[4].plot(simulation_time, vo, label="vo")
        # axs[4].set_xlabel("time (s)")
        # axs[4].set_ylabel("vo")
        # axs[4].legend()
        # axs[4].grid(True)

        # # Plot PWM
        # axs[5].plot(simulation_time, pwm_value, label="pwm")
        # axs[5].set_xlabel("time (s)")
        # axs[5].set_ylabel("pwm")
        # axs[5].legend()
        # axs[5].grid(True)

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


# fs = 1e6
# Ts = 1 / fs
# simulation_length_seconds = 1

# simulation_sample_length = int(simulation_length_seconds / Ts)

# input_voltage = []
# input_voltage.extend([20] * 250000)
# input_voltage.extend([10] * 750000)
# vref = []
# vref.extend([10] * 500000)
# vref.extend([5] * 500000)

# buck = buck_converter()
# buck.simulate(fs, simulation_length_seconds, input_voltage, Vref=vref)
# import matplotlib.pyplot as plt

# plt.show()
