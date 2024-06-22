import numpy as np
import voltage_mode_controller as vmc
import bode_plot as bp
from multiprocessing import Process, Queue, Pipe
from gui import gui

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
        output_current_limit=np.Inf,
        output_voltage_limit=np.Inf,
        inductor_current_limit=np.Inf,
        synchronous=False,
        controller=None,
    ) -> None:
        self.L = L
        self.Lesr = Lesr
        self.C = C
        self.Cesr = Cesr
        self.Rsource = Rsource
        self.Rload = Rload
        self.Vdiode = Vdiode
        self.Rdiode = Rdiode
        self.output_current_limit = output_current_limit
        self.inductor_current_limit = inductor_current_limit
        self.synchronous = synchronous
        if controller is None:
            controller = vmc.voltage_mode_controller()
        self.controller = controller

        # GUI App - currently associated per buck converter instance
        self.to_gui_queue = Queue()
        self.from_gui_queue = Queue()
        self.gui_process = Process(
            target=gui, args=(self.to_gui_queue, self.from_gui_queue)
        )
        self.gui_process.start()
        # TODO: eventually move all plotting to the simulator (ie. buck converter)
        # but this will require a more generalised approach to returning all plot variables and storing their history in the simulator
        # because it's too manual / requires too much effort to track new variables right now
        self.controller.to_gui_queue = self.to_gui_queue
        self.controller.from_gui_queue = self.from_gui_queue

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

        # model the freewheeling diode state for non-synchronous buck converters
        if not self.synchronous:
            # dcm occurs if il < 0 in off state
            if il < 0:
                il = 0
            vl = L * (il - il0) / PA
            # vc, vo, and il are unaffected by dcm

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
        pwm0,
        pwm_latch0,
        pwm_counter0,
        pulse_count0,
        saw_amp,
        Fpwm,
        Nskip,
        Ts,
    ):
        Tpwm = 1 / Fpwm

        vpwm = pwm0

        saw_increment = Ts / Tpwm * saw_amp
        saw = (saw0 + saw_increment) % saw_amp

        pwm_latch = pwm_latch0
        # clock pwm - first clock will be at t=0 + Tpwm
        if (saw0 + saw_increment) > saw_amp:
            # pwm_latch = 0
            if vcontrol > saw:
                vpwm = 1

        if vcontrol < saw:
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

        return saw, pwm_counter, pulse_count, vpwm, pwm_latch

    def comparator(self, input, reference):
        output = 0
        if input > reference:
            output = 1
        else:
            output = 0
        return output

    def analyse(self, RLOAD, bode_plant=True, bode_compensator=True):
        import bode_plot as bp

        """ TODO: because the converter parameters can be array (so that time varying parameters can be used in simulation), there should be some check to ensure none of the parameters are time varying, OR, need to just grab the first element of something. """

        L = self.L
        RL = self.Lesr
        C = self.C
        RC = self.Cesr
        RS = self.Rsource
        # RLOAD = self.Rload
        VD = self.Vdiode
        RD = self.Rdiode

        if bode_plant:
            # Zout (s):
            # 1. |Zc| = 1/wC
            # 2. |Zl| = wL
            # 3. f0 = 1 / (2*pi * sqrt(L*C))
            # 4. Q = R / R0
            # 5. R0 = sqrt(L / C)

            # Simplified buck converter to start with

            # vo / d
            # 1. No approximation, ZPK (zero, pole, gain) transfer function
            # Approach 1: Factor denominator
            # G(s) = K / [1 + a1*s + a2*s^2]
            # G(s) = K / [(1 - s/s1) * (1 - s/s2)]
            # s1 = -a1 / (2a2) * [1 - sqrt(1 - 4a2/[a1^2])]
            # s2 = -a1 / (2a2) * [1 + sqrt(1 - 4a2/[a1^2])]

            # for buck converter we get:
            # a1 = L/R, a2 = L*C
            a1 = L / RLOAD
            a2 = L * C
            print(a1)
            print(a2)

            # Check 4a2 <= a1^2, proceed, else, roots are complex (section 8.1.1 of fundamental of power electronics assumption is violated)
            if 4 * a2 > a1**2:
                # raise Exception(
                #     f"Error constructing transfer function, 4a2 > a1^2: {4*a2} > {a1**2}"
                # )
                # Complex roots
                s1 = -a1 / (2 * a2) * (1 - np.lib.scimath.sqrt(1 - 4 * a2 / (a1**2)))
                s2 = -a1 / (2 * a2) * (1 + np.lib.scimath.sqrt(1 - 4 * a2 / (a1**2)))
            else:
                s1 = -a1 / (2 * a2) * (1 - np.sqrt(1 - 4 * a2 / (a1**2)))
                s2 = -a1 / (2 * a2) * (1 + np.sqrt(1 - 4 * a2 / (a1**2)))
            print(f"s1: {s1}")
            print(f"s2: {s2}")

            f1 = -s1 / (2 * np.pi)
            f2 = -s2 / (2 * np.pi)
            bode = bp.bode_plot_zpk(1, f_poles=[f1, f2])
            bode.plot(0.1, 10e6)

            pass

        if bode_compensator:
            pass

        return True

    def simulate(
        self,
        fs,
        sim_length_seconds,
        Vin,
        pwm_frequency=10e3,
        pwm_Nskip=0,
        pwm_duty_cycle=None,
        Vref=0,
        vref_gain=1,
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
        if (
            isinstance(self.Rload, (list, np.ndarray))
            and len(self.Rload) == simulation_sample_length
        ):
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
            vref = [x * vref_gain for x in Vref]
        else:
            vref = [Vref * vref_gain] * simulation_sample_length
        # Pulse skipping PWM generator variables
        saw0 = 0
        saw = [saw0] * simulation_sample_length
        pwm_counter0 = 0
        pwm_counter = [pwm_counter0] * simulation_sample_length
        pulse_count0 = 0
        pulse_count = [pulse_count0] * simulation_sample_length
        pwm_latch0 = 0
        pwm_latch = [pwm_latch0] * simulation_sample_length
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
                # controller
                vcontrol[curr] = self.controller.simulate(vref[prev], vo[prev], dt)

                # TODO: should integrate the current limiting into the controller
                # TODO: combine many controllers into single controller would be nice
                if abs(il[prev]) > abs(self.inductor_current_limit):
                    vcontrol[curr] = 0
                if abs(io[prev]) > abs(self.output_current_limit):
                    vcontrol[curr] = 0

                # TODO: better pulse skipping modes
                if False:  # self.simple_pulse_skipping_mode:
                    # very simple pulse skipping mode method which could cause issues depending on buck converter parameters
                    if vo[prev] > 0.2 * vref[prev] and vo[prev] < 0.8 * vref[prev]:
                        vcontrol[curr] = 0

                # pwm generation

                (
                    saw[curr],
                    pwm_counter[curr],
                    pulse_count[curr],
                    pwm_value[curr],
                    pwm_latch[curr],
                ) = self.pulse_skipping_pwm_generator(
                    vcontrol[curr],
                    saw[prev],
                    pwm_value[prev],
                    pwm_latch[prev],
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
            verr[curr] = vref[curr] / vref_gain - vo[curr]

        if True:
            # Plot simulation
            import matplotlib.pyplot as plt
            import mplcursors

            plt.ioff()

            # Create the subplots
            fig, axs = plt.subplots(11, figsize=(10, 30), sharex=True)

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
            axs[i].plot(simulation_time, [x / vref_gain for x in vref], label="vref")
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
                axs[i].plot(simulation_time, pwm_value, label="vcontrol")
            axs[i].set_xlabel("time (s)")
            axs[i].set_ylabel("vcontrol (V)")
            axs[i].legend()
            axs[i].grid(True)

            i += 1
            axs[i].plot(simulation_time, pwm_value, label="vpwm")
            axs[i].set_xlabel("time (s)")
            axs[i].set_ylabel("vpwm (V)")
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
