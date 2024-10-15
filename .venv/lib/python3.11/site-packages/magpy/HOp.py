import numpy as np

from .methods import eye, is_square, kron

class HOp:
    """
    Represents a constant Hamiltonian operator. This can be for a system 
    containing one spin, for a single spin in part of a larger system, or for 
    an interactive component of a larger system.

    Attributes
    ----------
    data : ndarray
        Matrix representing the quantum operator.

    Examples
    --------
    One spin:
        H = sigmax

        >>> mp.HOp(mp.sigmax())
        >>> mp.HOp(1, 1, mp.sigmax())

    Two spins:
        H = sigmax x Id

        >>> mp.HOp(2, 1, mp.sigmax())

    Two spins interacting:
        H = sigmax x sigmay

        >>> mp.HOp(2, (1,mp.sigmax()), (2,mp.sigmay()))

    """

    def __init__(self, *args):
        """
        Construct matrix representing the quantum operator.
        """

        if not args:
            raise TypeError("input cannot be empty")

        if len(args) == 1 and is_square(args[0]):
            # single ndarray

            self.data = args[0]

        elif isinstance(args[1], tuple):
            # list of tuples (with pos and ndarray)

            matrices = args[0] * [eye(2)]

            for spin in args[1:]:
                matrices[spin[0] - 1] = spin[1]

            self.data = kron(matrices)
            
        elif args[0] >= args[1] and is_square(args[2]):
            # multi-spin system with one spin specified

            matrices = args[0] * [eye(2)]
            matrices[args[1] - 1] = args[2]
            self.data = kron(matrices)

        else:
            raise ValueError("invalid input")

    def __call__(self):
        return self.data

    def is_hermitian(self):
        return np.array_equal(self.data.conj().T, self.data)