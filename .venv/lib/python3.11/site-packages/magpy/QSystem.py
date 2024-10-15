import scipy.integrate

from .HOp import HOp
from .methods import liouvillian, commutator, vec, unvec

class QSystem:
    """
    Represents a quantum system with a specified Hamiltonian, allowing for the
    simulation of the system using a two-term Magnus expansion without 
    requiring re-calculation of the Liouvillians each time.

    Parameters
    ----------
    H : dict
        The Hamiltonian of the system, which takes the form:

            H = { f_1 : H_1, f_2 : H_2, ... },
        
        where each f_i is a function and each H_i is a HOp object. 

    Attributes
    ----------
    H1 : dict
        Pre-calculated Liouvillians and corresponding functional coefficients
        for the first term.

        Takes the form: 
            
            { f_1 : L(H_1), f_2 : L(H_2), ... },

        where f_i are functions (or constants) and L(H_i) are liouvillians of 
        the corresponding matrices. L(H_i) are square ndarrays.

    H2 : dict
        Pre-calculated Liouvillians and corresponding functional coefficients
        for the second term.

        Takes the form: 
        
            { f_1(y)*f_2(x) : L([H_1, H_2]), 
              f_1(y)*f_3(x) : L([H_1, H_3]), ... },
                
        where each entry is the two-variable function and its corresponding
        liouvillian of a commutator. L([H_i, H_j]) are square ndarrays.

    Examples
    --------
    Single spin system: 
        H(t) = f(t)*sigmax + g(t)*sigmay

        >>> H = {f : mp.HOp(1,1,mp.sigmax()), g : mp.HOp(1,1,mp.sigmay())}

    Two spin system: 
        H(t) = f(t)*(sigmax x Id) + g(t)*(Id x sigmay)

        >>> H = {f : mp.HOp(2,1,mp.sigmax()), g : mp.HOp(2,2,mp.sigmay())}

    Two spin system with repeated coefficient:
        H(t) = f(t)*(sigmax x Id) + f(t)*(Id x sigmay) + g(t)*(sigmaz x Id)

        >>> H = {f : [mp.HOp(2,1,mp.sigmax()), mp.HOp(2,2,mp.sigmay())], 
        g : mp.HOp(2,1,mp.sigmaz())}

    Interacting systems:
        H(t) = f(t)*(sigmax x Id) + g(t)*(sigmax x sigmay)

        >>> H = {f : mp.HOp(2,1,mp.sigmax()), 
        g : mp.HOp(2,(1,mp.sigmax()),(2,mp.sigmay()))}

        H(t) = f(t)*(Id x sigmax x Id) + g(t)*(sigmax x sigma x Id)

        >>> H = {f : mp.HOp(3,2,mp.sigmax()), 
                 g : mp.HOp(3,(1,mp.sigmax()),(2,mp.sigmax()))}

    """

    def __init__(self, H):
        """
        Perform pre-calculations for a two term Magnus expansion, and stored 
        in H1 and H2 to reduce reptitive calculations.

        Parameters
        ----------
        H : dict
            Hamiltonian of the system, which takes the form:

                H = { f_1 : H_1, f_2 : H_2, ... },
            
            where f_i are functions and H_i are HOp objects.

        """
        self.H1 = self.__setup_first_term(H)
        self.H2 = self.__setup_second_term(H)

    def update_hamiltonian(self, H):
        self.__init__(H)

    def __setup_first_term(self, H):
        """
        Pre-calculate Liouvillians for matrices (HOp) in H 
        for first term.
        """
        H1 = {}

        for coeff in H:
            if isinstance(H[coeff], HOp):
                # convert single HOp to list of (single) HOp
                H[coeff] = [H[coeff]]

            matrices = []
            for matrix in H[coeff]:
                matrices.append(liouvillian(matrix()))
            
            H1[coeff] = matrices
        
        return H1

    def __setup_second_term(self, H):
        """
        Pre-calculate Liouvillians for matrices (HOp) in H 
        for second term.
        """
        temp = []

        for coeff in H:
            if isinstance(H[coeff], list):
                for matrix in H[coeff]:
                    temp.append((coeff, matrix()))
            else:
                temp.append((coeff, H[coeff]()))

        n = len(temp)
        H2 = {}

        for i in range(n):
            for j in range(n):
                if i != j:
                    def f(y, x): return temp[i][0](y) * temp[j][0](x)

                    H2[f] = 0.5j * liouvillian(commutator(temp[i][1], 
                                                                temp[j][1]))
                                                                
        return H2

    def __eval_first_term(self, t0, tf):
        """
        Evaluate first term of Magnus expansion over specified time
        interval using pre-calculated terms.
        """
        total = 0

        for coeff in self.H1:
            if isinstance(coeff, (int, float)):
                for matrix in self.H1[coeff]:
                    total = total + matrix*coeff*(tf - t0)
            else:
                c = scipy.integrate.quad(coeff, t0, tf)[0]
                for matrix in self.H1[coeff]:
                    total = total + c*matrix
        
        return total

    def __eval_second_term(self, t0, tf):
        """
        Evaluate second term of Magnus expansion over specified time
        interval using pre-calculated terms.
        """
        total = 0
        def x(x): return x

        for coeff in self.H2:
            c = scipy.integrate.dblquad(coeff, t0, tf, t0, x)[0]
            total = total + c*self.H2[coeff]

        return total

    def lvn_solve(self, rho0, tlist):
        """
        Liouville-von Neumann evolution of density matrix for given 
        Hamiltonian.

        Parameters
        ----------
        rho0 : ndarray
            Initial density matrix.
        tlist : list / array
            Times at which to evaluate density matrices.

        Returns
        -------
        ndarray
            Density matrices calculated across tlist.

        Example
        -------
        >>> H = {f : mp.HOp(2,1,mp.sigmax())}
        >>> q_sys = mp.System(H)
        >>> rho0 = mp.HOp(2,1,mp.sigmay())
        >>> tlist = mp.linspace(0, 10, 0.5**5)
        >>> q_sys.lvn_solve(rho0, tlist)

        """
        states = [vec(rho0())]

        for i in range(len(tlist) - 1):

            omega = (self.__eval_first_term(tlist[i], tlist[i+1])
                     + self.__eval_second_term(tlist[i], tlist[i+1]))

            states.append(scipy.linalg.expm(omega) @ states[i])
            states[i] = unvec(states[i])
        
        states[-1] = unvec(states[-1])

        return states