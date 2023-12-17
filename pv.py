from scipy.optimize import fsolve
import numpy as np
import matplotlib.pyplot as plt
import mplcursors
from function_timer import FunctionTimer


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
        f1 = Iph - Io * (np.exp(Voc / (a * Ns * Vt)) - 1) - Voc / Rsh
        f2 = Isc - Iph + Io * (np.exp(Isc * Rs / (a * Ns * Vt)) - 1) + Isc * Rs / Rsh
        f3 = (
            Impp
            - Iph
            + Io * (np.exp((Vmpp + Impp * Rs) / (a * Ns * Vt)) - 1)
            + (Vmpp + Impp * Rs) / Rsh
        )
        f4 = Rs / Rsh - Io / (a * Ns * Vt) * (np.exp(Isc * Rs / (a * Ns * Vt))) * (
            Rsh - Rs
        )
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

        # Assign the solutions to the instance variables
        self.Iph = Iph
        self.Io = Io
        self.a = a
        self.Rs = Rs
        self.Rsh = Rsh

    def pv_current(self, V, I_0=0):
        """
        Calculate the current of the photovoltaic model.

        Parameters:
        V (float): The voltage.
        I_0 (float): The initial guess for the current. Default is 0.

        Returns:
        float: The current.
        """

        def current_eqn(I):
            return (
                self.Iph
                - self.Io
                * (np.exp((V + I * self.Rs) / (self.a * self.Ns * self.Vt)) - 1)
                - I
            )

        # Solve the equation for the current
        I = fsolve(current_eqn, I_0)
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

    def example(self):
        """
        Set the inputs to example values, calculate the parameters, and plot the results.
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

        self.plot()  # Call the plot method

    def plot(self):
        """
        Plot the V-I curve, V-P curve, V-dI/dV curve, and V-dP/dV curve of the photovoltaic model.
        """

        # Generate a range of voltage values
        V = np.linspace(0, self.Voc, 500)

        # Calculate the corresponding current values
        I = np.array([self.pv_current(v) for v in V])

        # Calculate power values
        P = np.array([self.pv_power(v) for v in V])

        # Calculate dI/dV and dP/dV values
        dI = np.array([self.pv_current_derivative(v) for v in V])
        dP = np.array([self.pv_power_derivative(v) for v in V])

        # Create the subplots
        fig, axs = plt.subplots(4, figsize=(10, 24), sharex=True)

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
        plt.show()

    def benchmark(self):
        func1 = lambda: self.pv_current(0)
        timer = FunctionTimer([func1], timeout=5)
        timer.time()


model = pv_model()
model.example()
model.benchmark()
