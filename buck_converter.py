import numpy as np
import voltage_mode_controller as vmc

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
        controller=vmc.voltage_mode_controller(),
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
        self.controller = controller

    def on_state(self, t0, PA, Vin, Rs, il0, Rl, L, vc0, ESR, C, R):
        K1 = C + ESR * C / R
        KA = -ESR * C / (K1 * L) - (Rl + Rs) / L
        KB = ESR * C / (R * K1 * L) - 1 / L
        vc = vc0 * (1 - PA / (R * K1)) + il0 * (PA / K1)
        il = Vin * PA / L + il0 * (1 + KA * PA) + vc0 * (KB * PA)
        ic = C * (vc - vc0) / PA
        vo = ESR * ic + vc
        vl = L * (il - il0) / PA
        t = t0 + PA
        return t, il, ic, vc, vo, vl

    def off_state(self, t0, PA, Vd, Rd, il0, Rl, L, vc0, ESR, C, R):
        K1 = C + ESR * C / R
        KA = -ESR * C / (K1 * L) - (Rl + Rd) / L
        KB = ESR * C / (R * K1 * L) - 1 / L
        vc = vc0 * (1 - PA / (R * K1)) + il0 * (PA / K1)
        il = -Vd * PA / L + il0 * (1 + KA * PA) + vc0 * (KB * PA)
        ic = C * (vc - vc0) / PA
        vo = ESR * ic + vc
        vl = L * (il - il0) / PA
        t = t0 + PA
        return t, il, ic, vc, vo, vl

    def PWM_generator(self, P, DC, Ts, TAM):
        import numpy as np

        PA = Ts
        PWM = np.zeros(TAM)
        t = np.zeros(TAM)
        if DC == 1:
            PWM = np.ones(TAM)
            return t, PWM
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

    def pulse_skipping_pwm_generator(
        self,
        vcontrol,
        saw0,
        pwm_counter0,
        pulse_count0,
        saw_amp,
        Fpwm,
        Nskip,
        Ts,
    ):
        Tpwm = 1 / Fpwm
        saw_increment = Ts / Tpwm * saw_amp
        saw = (saw0 + saw_increment) % saw_amp

        if vcontrol > saw:
            vpwm = 1
        else:
            vpwm = 0

        pwm_counter = pwm_counter0
        pulse_count = pulse_count0
        if pwm_counter == 0 and Nskip != 0:
            # mod Nskip for more meaningful plots of pulse_count
            pulse_count = (pulse_count + 1) % Nskip
        pwm_counter = int((pwm_counter + 1) % (Tpwm / Ts))

        if Nskip != 0 and pulse_count != 0:
            # skip pulses
            vpwm = 0

        return saw, pwm_counter, pulse_count, vpwm

    def comparator(self, input, reference):
        output = 0
        if input > reference:
            output = 1
        else:
            output = 0
        return output

    def simulate(
        self,
        fs,
        sim_length_seconds,
        Vin,
        pwm_frequency=10e3,
        pwm_Nskip=0,
        pwm_duty_cycle=None,
        Vref=0,
        output_current_limit=np.Inf,
        output_voltage_limit=np.Inf,
    ):
        Ts = 1 / fs
        PA = Ts

        simulation_sample_length = int(sim_length_seconds / Ts)

        # Load stored parameters to local variables
        L = self.L
        Lesr = self.Lesr
        C = self.C
        Cesr = self.Cesr
        Rsource = self.Rsource
        if len(self.Rload) == simulation_sample_length:
            Rload = self.Rload
        else:
            Rload = [self.Rload] * simulation_sample_length
        Vdiode = self.Vdiode
        Rdiode = self.Rdiode

        if pwm_duty_cycle is not None:
            # Fixed PWM generation
            pwm_period = 1 / pwm_frequency
            pwm_time, pwm_value = self.PWM_generator(
                pwm_period, pwm_duty_cycle, Ts, simulation_sample_length
            )
        else:
            pwm_value = [0] * simulation_sample_length
            # Fixed sawtooth generation
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
        io = [0] * simulation_sample_length
        vl = [0] * simulation_sample_length
        if len(Vin) == simulation_sample_length:
            vin = Vin
        else:
            vin = [Vin] * simulation_sample_length
        # Controller Variables
        if pwm_duty_cycle is None:
            vcontrol0 = 0
            vcontrol = [vcontrol0] * simulation_sample_length
            vcf0 = 0
            vcf = [vcf0] * simulation_sample_length
        if len(Vref) == simulation_sample_length:
            vref = Vref
        else:
            vref = [Vref] * simulation_sample_length
        # Pulse skipping PWM generator variables
        saw0 = 0
        saw = [saw0] * simulation_sample_length
        pwm_counter0 = 0
        pwm_counter = [pwm_counter0] * simulation_sample_length
        pulse_count0 = 0
        pulse_count = [pulse_count0] * simulation_sample_length
        # vpwm0 = 0
        # vpwm = [vpwm0] * simulation_sample_length
        # Extra Variables
        verr = [vref[0] - vo[0]] * simulation_sample_length
        # Simulation time
        simulation_time = [0] * simulation_sample_length
        # Simulation loop
        simulation_sample_time = 0
        while simulation_sample_time + 1 < simulation_sample_length:
            simulation_sample_time += 1

            # simulation sample time naming convention
            curr = simulation_sample_time
            prev = simulation_sample_time - 1

            simulation_time[curr] = simulation_time[prev] + Ts
            dt = Ts

            if pwm_duty_cycle is None:
                vcontrol[curr] = self.controller.simulate(vref[prev], vo[prev], dt)

                (
                    saw[curr],
                    pwm_counter[curr],
                    pulse_count[curr],
                    pwm_value[curr],
                ) = self.pulse_skipping_pwm_generator(
                    vcontrol[curr],
                    saw[prev],
                    pwm_counter[prev],
                    pulse_count[prev],
                    1,
                    pwm_frequency,
                    pwm_Nskip,
                    Ts,
                )
                # pwm_value[curr] = self.comparator(
                #     vcontrol[curr],
                #     sawtooth_value[curr],
                # )

            # Output current limiting
            # if io[prev] > output_current_limit:
            #     pwm_value[curr] = 0

            # # Output voltage limiting
            # if vo[prev] > output_voltage_limit:
            #     pwm_value[curr] = 0

            if pwm_value[curr]:
                # On state
                (
                    discard,
                    il[curr],
                    ic[curr],
                    vc[curr],
                    vo[curr],
                    vl[curr],
                ) = self.on_state(
                    simulation_time[curr],
                    PA,
                    vin[curr],
                    Rsource,
                    il[prev],
                    Lesr,
                    L,
                    vc[prev],
                    Cesr,
                    C,
                    Rload[curr],
                )
            else:
                # Off state
                (
                    discard,
                    il[curr],
                    ic[curr],
                    vc[curr],
                    vo[curr],
                    vl[curr],
                ) = self.off_state(
                    simulation_time[curr],
                    PA,
                    Vdiode,
                    Rdiode,
                    il[prev],
                    Lesr,
                    L,
                    vc[prev],
                    Cesr,
                    C,
                    Rload[curr],
                )

            io[curr] = vo[curr] / Rload[curr]
            verr[curr] = vref[curr] - vo[curr]

        # Plot simulation
        import matplotlib.pyplot as plt
        import mplcursors

        # Create the subplots
        fig, axs = plt.subplots(10, figsize=(10, 30), sharex=True)

        i = 0
        # Plot Input Voltage
        axs[i].plot(simulation_time, vin, label="vin")
        axs[i].set_xlabel("time (s)")
        axs[i].set_ylabel("vin (V)")
        axs[i].legend()
        axs[i].grid(True)

        i += 1
        # Plot inductor current
        axs[i].plot(simulation_time, il, label="il")
        axs[i].set_xlabel("time (s)")
        axs[i].set_ylabel("il (A)")
        axs[i].legend()
        axs[i].grid(True)

        i += 1
        # Plot inductor voltage
        axs[i].plot(simulation_time, vl, label="vl")
        axs[i].set_xlabel("time (s)")
        axs[i].set_ylabel("vl (V)")
        axs[i].legend()
        axs[i].grid(True)

        i += 1
        # Plot capacitor voltage
        axs[i].plot(simulation_time, vc, label="vc")
        axs[i].set_xlabel("time (s)")
        axs[i].set_ylabel("vc (V)")
        axs[i].legend()
        axs[i].grid(True)

        i += 1
        # Plot Reference Voltage
        axs[i].plot(simulation_time, vref, label="vref")
        axs[i].set_xlabel("time (s)")
        axs[i].set_ylabel("vref (V)")
        axs[i].legend()
        axs[i].grid(True)

        # Plot Output Voltage
        axs[i].plot(simulation_time, vo, label="vout")
        axs[i].set_xlabel("time (s)")
        axs[i].set_ylabel("vout (V)")
        axs[i].legend()
        axs[i].grid(True)

        i += 1
        # Plot Output Current
        axs[i].plot(simulation_time, io, label="iout")
        axs[i].set_xlabel("time (s)")
        axs[i].set_ylabel("iout (A)")
        axs[i].legend()
        axs[i].grid(True)

        i += 1
        # Plot Rload
        axs[i].plot(simulation_time, Rload, label="rload")
        axs[i].set_xlabel("time (s)")
        axs[i].set_ylabel("rload (ohms)")
        axs[i].legend()
        axs[i].grid(True)

        i += 1
        # Plot Error Voltage
        axs[i].plot(simulation_time, verr, label="verr")
        axs[i].set_xlabel("time (s)")
        axs[i].set_ylabel("verr (V)")
        axs[i].legend()
        axs[i].grid(True)

        i += 1
        # PWM generation internals
        axs[i].plot(simulation_time, saw, label="saw")
        # axs[i].plot(simulation_time, pwm_counter, label="pwm_counter")
        axs[i].plot(simulation_time, pulse_count, label="pulse_count")
        axs[i].set_xlabel("time (s)")
        # axs[i].set_ylabel("verr (V)")
        axs[i].legend()
        axs[i].grid(True)

        i += 1
        # Plot Control Voltage
        if pwm_duty_cycle is None:
            axs[i].plot(simulation_time, vcontrol, label="vcontrol")
            # axs[i].plot(simulation_time, pwm_value, label="pwm")
        else:
            axs[i].plot(
                simulation_time,
                pwm_value,
                label="vcontrol",
            )
        axs[i].set_xlabel("time (s)")
        axs[i].set_ylabel("vcontrol (V)")
        axs[i].legend()
        axs[i].grid(True)

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
