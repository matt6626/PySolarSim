from scipy.optimize import fsolve, brentq
from scipy.special import lambertw
import numpy as np
import matplotlib.pyplot as plt
import mplcursors
from function_timer import FunctionTimer

# At some point, this paper may be of interest: https://arxiv.org/pdf/1601.02679.pdf


class pv_model:
    def __init__(self, Isc=None, Voc=None, Impp=None, Vmpp=None, Ns=None) -> None:
        """
        Initialize the model with given parameters.

        Parameters:
        Isc (float): Short-circuit current. Default is None.
        Voc (float): Open-circuit voltage. Default is None.
        Impp (float): Current at maximum power point. Default is None.
        Vmpp (float): Voltage at maximum power point. Default is None.
        Ns (int): Number of cells in series. Default is None.
        """

        # Assign the input parameters to instance variables
        self._Isc = Isc
        self._Voc = Voc
        self._Impp = Impp
        self._Vmpp = Vmpp
        self._Ns = Ns

        # Define constants
        self.k = 1.38 * 10**-23  # Boltzmann constant
        self.q = 1.60 * 10**-19  # Elementary charge
        self.Tstc = 25 + 273.15  # Standard test conditions temperature in Kelvin
        self.Vt = self.k / self.q * self.Tstc  # Thermal voltage

        # Calculate the model parameters
        self.calculate_params()

    @property
    def Isc(self):
        """Getter for the short-circuit current (Isc)."""
        return self._Isc

    @Isc.setter
    def Isc(self, value):
        """
        Setter for the short-circuit current (Isc).
        Recalculates model parameters when Isc is set.
        """
        self._Isc = value
        self.calculate_params()

    @property
    def Voc(self):
        """Getter for the open-circuit voltage (Voc)."""
        return self._Voc

    @Voc.setter
    def Voc(self, value):
        """
        Setter for the open-circuit voltage (Voc).
        Recalculates model parameters when Voc is set.
        """
        self._Voc = value
        self.calculate_params()

    @property
    def Impp(self):
        """Getter for the current at maximum power point (Impp)."""
        return self._Impp

    @Impp.setter
    def Impp(self, value):
        """
        Setter for the current at maximum power point (Impp).
        Recalculates model parameters when Impp is set.
        """
        self._Impp = value
        self.calculate_params()

    @property
    def Vmpp(self):
        """Getter for the voltage at maximum power point (Vmpp)."""
        return self._Vmpp

    @Vmpp.setter
    def Vmpp(self, value):
        """
        Setter for the voltage at maximum power point (Vmpp).
        Recalculates model parameters when Vmpp is set.
        """
        self._Vmpp = value
        self.calculate_params()

    @property
    def Ns(self):
        """Getter for the number of cells in series (Ns)."""
        return self._Ns

    @Ns.setter
    def Ns(self, value):
        """
        Setter for the number of cells in series (Ns).
        Recalculates model parameters when Ns is set.
        """
        self._Ns = value
        self.calculate_params()

    def check_inputs(self):
        """
        Checks if all inputs are not None.

        Returns:
        bool: True if all inputs are not None, False otherwise.
        """
        inputs = [self.Isc, self.Voc, self.Impp, self.Vmpp, self.Ns]
        if all(input is not None for input in inputs):
            return True
        else:
            return False

    def equations(self, vars):
        """
        Define the system of equations for the photovoltaic model.

        Parameters:
        vars (tuple): A tuple containing the variables of the system.

        Returns:
        list: A list of equations.
        """
        Iph, Io, a, Rs, Rsh = vars
        Voc = self.Voc
        Ns = self.Ns
        Vt = self.Vt
        Isc = self.Isc
        Impp = self.Impp
        Vmpp = self.Vmpp

        # Define the equations
        # f1: Condition at open circuit (I=0)
        f1 = Iph - Io * (np.exp(Voc / (a * Ns * Vt)) - 1) - Voc / Rsh
        # f2: Condition at short circuit (V=0)
        f2 = Isc - Iph + Io * (np.exp(Isc * Rs / (a * Ns * Vt)) - 1) + Isc * Rs / Rsh
        # f3: Condition at maximum power point
        f3 = (
            Impp
            - Iph
            + Io * (np.exp((Vmpp + Impp * Rs) / (a * Ns * Vt)) - 1)
            + (Vmpp + Impp * Rs) / Rsh
        )
        # f4: Derivative of current with respect to voltage should be zero at maximum power point
        f4 = Rs / Rsh - Io / (a * Ns * Vt) * (np.exp(Isc * Rs / (a * Ns * Vt))) * (
            Rsh - Rs
        )
        # f5: Derivative of power with respect to voltage should be zero at maximum power point
        f5 = Impp - (Vmpp - Impp * Rs) * (
            Io / (a * Ns * Vt) * (np.exp((Vmpp + Impp * Rs) / (a * Ns * Vt))) + 1 / Rsh
        )

        return [f1, f2, f3, f4, f5]

    def backcalculate_equations(self, vars):
        """
        Define the system of equations to back calculate the original parameters.

        Parameters:
        vars (tuple): A tuple containing the variables of the system.

        Returns:
        list: A list of equations.
        """
        Voc, Isc, Vmpp, Impp, Ns = vars
        Iph = self.Iph
        Io = self.Io
        a = self.a
        Rs = self.Rs
        Rsh = self.Rsh
        Vt = self.Vt

        # Define the equations
        # f1: Condition at open circuit (I=0)
        f1 = Iph - Io * (np.exp(Voc / (a * Ns * Vt)) - 1) - Voc / Rsh
        # f2: Condition at short circuit (V=0)
        f2 = Isc - Iph + Io * (np.exp(Isc * Rs / (a * Ns * Vt)) - 1) + Isc * Rs / Rsh
        # f3: Condition at maximum power point
        f3 = (
            Impp
            - Iph
            + Io * (np.exp((Vmpp + Impp * Rs) / (a * Ns * Vt)) - 1)
            + (Vmpp + Impp * Rs) / Rsh
        )
        # f4: Derivative of current with respect to voltage should be zero at maximum power point
        f4 = Rs / Rsh - Io / (a * Ns * Vt) * (np.exp(Isc * Rs / (a * Ns * Vt))) * (
            Rsh - Rs
        )
        # f5: Derivative of power with respect to voltage should be zero at maximum power point
        f5 = Impp - (Vmpp - Impp * Rs) * (
            Io / (a * Ns * Vt) * (np.exp((Vmpp + Impp * Rs) / (a * Ns * Vt))) + 1 / Rsh
        )

        return [f1, f2, f3, f4, f5]

    def calculate_params(self):
        """
        Calculate the parameters of the photovoltaic model.

        If not all inputs are set, print a message and return.
        """
        if not self.check_inputs():
            print("Some inputs are not set.")
            return

        # Initial guesses for the parameters
        Iph_0 = self.Isc
        a_0 = (2 * self.Vmpp - self.Voc) / (
            self.Ns
            * self.Vt
            * (self.Isc / (self.Isc - self.Impp) + np.log(1 - self.Impp / self.Isc))
        )
        Io_0 = self.Isc * np.exp(-self.Voc / (a_0 * self.Ns * self.Vt))
        Rs_0 = (
            a_0 * self.Ns * self.Vt * np.log(1 - self.Impp / self.Isc)
            + self.Voc
            - self.Vmpp
        ) / self.Impp
        Rsh_0 = self.Voc / (
            self.Isc - Io_0 * np.exp((self.Isc * Rs_0) / (a_0 * self.Ns * self.Vt))
        )

        # Solve the system of equations
        Iph, Io, a, Rs, Rsh = fsolve(self.equations, (Iph_0, Io_0, a_0, Rs_0, Rsh_0))

        # Plug the solutions back into the equations
        residuals = self.equations((Iph, Io, a, Rs, Rsh))

        # Print the residuals
        print(f"Residuals: {residuals}")

        # Assign the solutions to the instance variables
        self.Iph = Iph
        self.Io = Io
        self.a = a
        self.Rs = Rs
        self.Rsh = Rsh

        # Back calculate the input parameters
        self.backcalculate_params()

    def backcalculate_params(self):
        """
        Back calculate the original parameters of the photovoltaic model.

        If not all inputs are set, print a message and return.
        """
        if not self.check_inputs():
            print("Some inputs are not set.")
            return

        # Initial guesses for the parameters
        Voc_0 = self.Voc
        Isc_0 = self.Isc
        Vmpp_0 = self.Vmpp
        Impp_0 = self.Impp
        Ns_0 = self.Ns

        # Solve the system of equations
        self.Voc_bc, self.Isc_bc, self.Vmpp_bc, self.Impp_bc, self.Ns_bc = fsolve(
            self.backcalculate_equations, (Voc_0, Isc_0, Vmpp_0, Impp_0, Ns_0)
        )

        # Print the back-calculated parameters
        print(f"Back-calculated Voc: {self.Voc_bc}")
        print(f"Back-calculated Isc: {self.Isc_bc}")
        print(f"Back-calculated Vmpp: {self.Vmpp_bc}")
        print(f"Back-calculated Impp: {self.Impp_bc}")
        print(f"Back-calculated Ns: {self.Ns_bc}")

        # Print the original parameters
        print(f"Original Voc: {self.Voc}")
        print(f"Original Isc: {self.Isc}")
        print(f"Original Vmpp: {self.Vmpp}")
        print(f"Original Impp: {self.Impp}")
        print(f"Original Ns: {self.Ns}")

        # Calculate and print the error
        error_Voc = abs((self.Voc_bc - self.Voc) / self.Voc)
        error_Isc = abs((self.Isc_bc - self.Isc) / self.Isc)
        error_Vmpp = abs((self.Vmpp_bc - self.Vmpp) / self.Vmpp)
        error_Impp = abs((self.Impp_bc - self.Impp) / self.Impp)
        error_Ns = abs((self.Ns_bc - self.Ns) / self.Ns)

        print(f"Error in Voc: {error_Voc}")
        print(f"Error in Isc: {error_Isc}")
        print(f"Error in Vmpp: {error_Vmpp}")
        print(f"Error in Impp: {error_Impp}")
        print(f"Error in Ns: {error_Ns}")

    def pv_current(self, V, I_0=0, explicit=False):
        """
        Calculate the current of the photovoltaic model.

        Parameters:
        V (float): The voltage.
        I_0 (float): The initial guess for the current. Default is 0.

        Returns:
        float: The current.
        """

        if explicit:
            lambertw_arg = (
                (self.Rs * self.Io * self.Rsh)
                / ((self.a * self.Ns * self.Vt) * (self.Rs + self.Rsh))
                * np.exp(
                    (self.Rsh * (self.Rs * self.Iph + self.Rs * self.Io + V))
                    / (self.a * self.Ns * self.Vt * (self.Rs + self.Rsh))
                )
            )
            I = (
                -V / (self.Rs + self.Rsh)
                - lambertw(lambertw_arg) * (self.a * self.Ns * self.Vt) / self.Rs
                + self.Rsh * (self.Io + self.Iph) / (self.Rs + self.Rsh)
            )
            return np.real(I)

        else:

            def current_eqn(I):
                return (
                    self.Iph
                    - self.Io
                    * (np.exp((V + I * self.Rs) / (self.a * self.Ns * self.Vt)) - 1)
                    - ((V + I * self.Rs) / self.Rsh)
                    - I
                )

            # # Clamp the current to 0 at Voc
            # # or look at ways to fix the fsolve to return much smaller currents
            # # as it currently caps at 12mA for Voc
            # if V >= self.Voc_bc:
            #     I = 0
            #     I = np.array([I], np.float64)
            #     return I

            # Solve the equation for the current
            # I = fsolve(current_eqn, I_0)
            I = brentq(current_eqn, 0, self.Isc_bc)
            return I

    def pv_current_RL(self, R_L, I_0=0, explicit=False):
        """
        Calculate the current of the photovoltaic model.

        Parameters:
        R_L (float): The load resistance.
        I_0 (float): The initial guess for the current. Default is 0.

        Returns:
        float: The current.
        """

        if explicit:

            def current_eqn(I):
                V = I * R_L
                lambertw_arg = (
                    (self.Rs * self.Io * self.Rsh)
                    / ((self.a * self.Ns * self.Vt) * (self.Rs + self.Rsh))
                    * np.exp(
                        (self.Rsh * (self.Rs * self.Iph + self.Rs * self.Io + V))
                        / (self.a * self.Ns * self.Vt * (self.Rs + self.Rsh))
                    )
                )
                if lambertw_arg == np.inf:
                    lambertw_approx = (
                        np.log(
                            (self.Rs * self.Io * self.Rsh)
                            / ((self.a * self.Ns * self.Vt) * (self.Rs + self.Rsh))
                        )
                        + (
                            (self.Rsh * (self.Rs * self.Iph + self.Rs * self.Io + V))
                            / (self.a * self.Ns * self.Vt * (self.Rs + self.Rsh))
                        )
                        - np.log(
                            np.log(
                                (self.Rs * self.Io * self.Rsh)
                                / ((self.a * self.Ns * self.Vt) * (self.Rs + self.Rsh))
                            )
                            + (
                                (
                                    self.Rsh
                                    * (self.Rs * self.Iph + self.Rs * self.Io + V)
                                )
                                / (self.a * self.Ns * self.Vt * (self.Rs + self.Rsh))
                            )
                        )
                    )
                    return (
                        -V / (self.Rs + self.Rsh)
                        - lambertw_approx * (self.a * self.Ns * self.Vt) / self.Rs
                        + self.Rsh * (self.Io + self.Iph) / (self.Rs + self.Rsh)
                        - I
                    )
                else:
                    return (
                        -V / (self.Rs + self.Rsh)
                        - lambertw(lambertw_arg)
                        * (self.a * self.Ns * self.Vt)
                        / self.Rs
                        + self.Rsh * (self.Io + self.Iph) / (self.Rs + self.Rsh)
                        - I
                    )

            I = brentq(current_eqn, 0, self.Isc_bc)
            return I
        else:

            def current_eqn(I):
                V = I * R_L  # Calculate the voltage across the load
                V = np.clip(V, 0, self.Voc_bc)  # Clamp V within [0, Voc]

                return (
                    self.Iph
                    - self.Io
                    * (np.exp((V + I * self.Rs) / (self.a * self.Ns * self.Vt)) - 1)
                    - ((V + I * self.Rs) / self.Rsh)
                    - I
                )

            # Solve the equation for the current
            # I = fsolve(current_eqn, I_0)
            I = brentq(current_eqn, 0, self.Isc_bc)

            # if I * R_L >= self.Voc_bc:
            #     I = self.Voc / R_L
            #     I = np.array([I], np.float64)
            #     return I

            return I

    def pv_current_derivative(self, V):
        """
        Calculate the derivative of the current with respect to the voltage.

        Parameters:
        V (float): The voltage.

        Returns:
        float: The derivative of the current.
        """
        I = self.pv_current(V)
        dI = -(
            1
            + (self.Rsh * self.Io)
            / (self.a * self.Ns * self.Vt)
            * np.exp((V + I * self.Rs) / (self.a * self.Ns * self.Vt))
        ) / (
            self.Rsh
            + (self.Rs * self.Rsh * self.Io)
            / (self.a * self.Ns * self.Vt)
            * np.exp((V + I * self.Rs) / (self.a * self.Ns * self.Vt))
            + self.Rs
        )
        return dI

    def pv_power_derivative(self, V):
        """
        Calculate the derivative of the power with respect to the voltage.

        Parameters:
        V (float): The voltage.

        Returns:
        float: The derivative of the power.
        """
        I = self.pv_current(V)
        dP = (
            -V
            * (
                1
                + (self.Rsh * self.Io)
                / (self.a * self.Ns * self.Vt)
                * np.exp((V + I * self.Rs) / (self.a * self.Ns * self.Vt))
            )
            / (
                self.Rsh
                + (self.Rs * self.Rsh * self.Io)
                / (self.a * self.Ns * self.Vt)
                * np.exp((V + I * self.Rs) / (self.a * self.Ns * self.Vt))
                + self.Rs
            )
            + I
        )
        return dP

    def pv_power(self, V):
        """
        Calculate the power of the photovoltaic model.

        Parameters:
        V (float): The voltage.

        Returns:
        float: The power.
        """
        I = self.pv_current(V)
        P = V * I
        return P

    def pv_voltage(self, I, V_0=0, explicit=False):
        """
        Calculate the voltage of the photovoltaic model.

        Parameters:
        I (float): The current.
        V_0 (float): The initial guess for the voltage. Default is 0.

        Returns:
        float: The current.
        """

        if explicit:
            lambertw_arg = (
                self.Io
                * self.Rsh
                / (self.a * self.Ns * self.Vt)
                * np.exp(
                    (self.Rsh * (-I + self.Iph + self.Io))
                    / (self.a * self.Ns * self.Vt)
                )
            )
            V = (
                -I * self.Rs
                - I * self.Rsh
                + self.Iph * self.Rsh
                - self.a * self.Ns * self.Vt * lambertw(lambertw_arg)
                + self.Io * self.Rsh
            )

            if lambertw_arg == np.inf:
                # https://math.stackexchange.com/questions/3432288/avoid-arithmetic-overflow-when-calculating-lambertwexpx
                lambertw_approx = (
                    np.log(self.Io * self.Rsh / (self.a * self.Ns * self.Vt))
                    + (self.Rsh * (-I + self.Iph + self.Io))
                    / (self.a * self.Ns * self.Vt)
                    - np.log(
                        np.log(self.Io * self.Rsh / (self.a * self.Ns * self.Vt))
                        + (self.Rsh * (-I + self.Iph + self.Io))
                        / (self.a * self.Ns * self.Vt)
                    )
                )
                V = (
                    -I * self.Rs
                    - I * self.Rsh
                    + self.Iph * self.Rsh
                    - self.a * self.Ns * self.Vt * (lambertw_approx)
                    + self.Io * self.Rsh
                )

            return np.real(V)
        else:

            I = np.clip(I, 0, self.Isc)

            def voltage_eqn(V):
                # V = np.clip(V, 0, self.Voc_bc)  # Clamp V within [0, Voc]

                return (
                    self.Iph
                    - self.Io
                    * (np.exp((V + I * self.Rs) / (self.a * self.Ns * self.Vt)) - 1)
                    - ((V + I * self.Rs) / self.Rsh)
                    - I
                )

            # # Clamp the current to 0 at Voc
            # # or look at ways to fix the fsolve to return much smaller currents
            # # as it currently caps at 12mA for Voc
            # if V >= self.Voc_bc:
            #     I = 0
            #     I = np.array([I], np.float64)
            #     return I

            # Solve the equation for the current
            # V = fsolve(voltage_eqn, V_0)
            V = brentq(voltage_eqn, 0, self.Voc)
            return V

    def example(self):
        """
        Set the inputs to example values, and calculate the parameters.
        """
        self._Isc = 7.36
        self._Voc = 30.4
        self._Impp = 6.83
        self._Vmpp = 24.2
        self._Ns = 50
        self.calculate_params()

        print(f"Iph: {self.Iph}")
        print(f"Io: {self.Io}")
        print(f"a: {self.a}")
        print(f"Rs: {self.Rs}")
        print(f"Rsh: {self.Rsh}")

        print(f"I(V=Voc)={self.pv_current(self.Voc)}")
        print(f"I(V=Voc)={self.pv_current(self.Voc_bc)}")
        # print(f"V(I=Isc)={self.pv_current(self.Voc)}")
        # print(f"V(I=Isc)={self.pv_current(self.Voc_bc)}")

    def plot(self, explicit=False):
        """
        Plot the V-I curve, V-P curve, V-dI/dV curve, V-dP/dV curve, and R-V curve of the photovoltaic model.
        """

        # Generate a range of voltage values
        V = np.linspace(0, self.Voc, 500)

        # Calculate the corresponding current values
        I = np.array([self.pv_current(v, explicit=explicit) for v in V])

        # Calculate power values
        P = np.array([self.pv_power(v) for v in V])

        # Calculate dI/dV and dP/dV values
        dI = np.array([self.pv_current_derivative(v) for v in V])
        dP = np.array([self.pv_power_derivative(v) for v in V])

        # Generate a range of load resistance values
        R_L = np.logspace(-6, 6, 100)  # Avoid division by zero

        # Calculate the corresponding current values for the load resistance
        I_RL = np.array([self.pv_current_RL(r, 7, explicit=True) for r in R_L])

        # Calculate voltage values for the load resistance
        V_RL = np.squeeze(I_RL) * R_L  # Ohm's law
        # print("I_RL:")
        # print(I_RL)
        # print("V_RL:")
        # print(V_RL)
        # print("R_L:")
        # print(R_L)

        # Create the subplots
        fig, axs = plt.subplots(5, figsize=(10, 30), sharex=True)

        # Plot V-I curve
        axs[0].plot(V, I, label="V-I curve")
        axs[0].set_xlabel("Voltage (V)")
        axs[0].set_ylabel("Current (I)")
        axs[0].legend()
        axs[0].grid(True)

        # Plot V-P curve
        axs[1].plot(V, P, label="V-P curve")
        axs[1].set_xlabel("Voltage (V)")
        axs[1].set_ylabel("Power (P)")
        axs[1].legend()
        axs[1].grid(True)

        # Plot V-dI/dV curve
        axs[2].plot(V, dI, label="V-dI/dV curve")
        axs[2].set_xlabel("Voltage (V)")
        axs[2].set_ylabel("dI/dV")
        axs[2].legend()
        axs[2].grid(True)

        # Plot V-dP/dV curve
        axs[3].plot(V, dP, label="V-dP/dV curve")
        axs[3].set_xlabel("Voltage (V)")
        axs[3].set_ylabel("dP/dV")
        axs[3].legend()
        axs[3].grid(True)

        # Plot R-V curve
        axs[4].plot(V_RL, R_L, label="V-R curve")
        axs[4].set_xlabel("Voltage (V)")
        axs[4].set_ylabel("Resistance (R_L)")
        axs[4].legend()
        axs[4].grid(True)
        axs[4].set_yscale("log", base=10)

        # Add a linked cursor
        cursor = mplcursors.cursor(axs, hover=True)

        # Define the annotation for the cursor
        @cursor.connect("add")
        def on_add(sel):
            x, y = sel.target
            sel.annotation.set_text(f"x: {x:.2f}\ny: {y:.2f}")

        # Add a title to the figure
        fig.suptitle("PV Curves")

        # Display the plot
        plt.show(block=False)

    def benchmark(self):
        func1 = lambda: self.pv_current(0)
        timer = FunctionTimer([func1], timeout=5)
        timer.time()


model = pv_model()
model.example()
model.plot()
model.plot(explicit=True)
plt.show()

# # model.benchmark()
# print(f"model.pv_current(model.Voc): {model.pv_current(model.Voc)}")
# print(f"model.pv_current(0): {model.pv_current(0)}")
# print(f"model.pv_current(model.Voc_bc): {model.pv_current(model.Voc_bc)}")
# print(
#     f"model.pv_current_explicit(model.Voc): {model.pv_current_explicit(model.Voc, explicit=True)}"
# )
# print(f"model.pv_current_explicit(0): {model.pv_current_explicit(0, explicit=True)}")
# print(
#     f"model.pv_current_explicit(model.Voc_bc): {model.pv_current_explicit(model.Voc_bc, explicit=True)}"
# )
# print(model.Isc_bc)

# print(f"asdf: {model.pv_current(model.Voc + 1)}")

# print(f"voltage (I=7): {model.pv_voltage(7)}")
# print(f"voltage (I=8): {model.pv_voltage(8)}")
# print(f"voltage (I=1): {model.pv_voltage(1)}")
# print(f"voltage (I=0.5): {model.pv_voltage(0.5)}")
# print(f"voltage (I=0.1): {model.pv_voltage(0.1)}")
# print(f"voltage (I=0): {model.pv_voltage(0.001)}")
# print(f"current (V=0.001): {model.pv_current(model.pv_voltage(0.001))}")

# print(f"voltage_explicit (I=7): {model.pv_voltage(7, explicit=True)}")
# print(f"voltage_explicit (I=8): {model.pv_voltage(8, explicit=True)}")
# print(f"voltage_explicit (I=1): {model.pv_voltage(1, explicit=True)}")
# print(f"voltage_explicit (I=0.5): {model.pv_voltage(0.5, explicit=True)}")
# print(f"voltage_explicit (I=0.1): {model.pv_voltage(0.1, explicit=True)}")
# print(f"voltage_explicit (I=0): {model.pv_voltage(0.001, explicit=True)}")
# print(
#     f"current_explicit (V=0.001): {model.pv_current(model.pv_voltage(0.001, explicit=True), explicit=True)}"
# )
