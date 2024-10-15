# Copyright (c) 2021-2022 Danny Goodacre and Pranav Singh
# Use of this source code is governed by the MIT license that can be found in 
# the LICENSE file.

import numpy as np

def sigmax():
    return np.array([[0, 1], [1, 0]])


def sigmay():
    return np.array([[0, -1j], [1j, 0]])


def sigmaz():
    return np.array([[1, 0], [0, -1]])


def eye(n):
    return np.eye(n)


def vec(arr):
    """
    Return vectorised form of input using column-major (Fortran) ordering.

    Parameters
    ----------
    mat : ndarray
        Array.

    Returns
    -------
    ndarray
        Vector.

    """

    return np.asarray(arr).flatten('F')


def unvec(vec, c=None):
    """
    Return unvectorised vector using column-major (Fortran) ordering.

    Parameters
    ----------
    vec : ndarray
        Vector of elements.
    c : int, optional
        Desired length of columns in array. Infers square array if possible.
        The default is None.

    Returns
    -------
    ndarray
        Array.

    """
    vec = np.array(vec)

    if (len(vec) % 2 != 0):
        # odd number of elements
        if (len(vec) == 1):
            return vec
        else:
            print("Error: odd number of elements in vector. \
                  Cannot form matrix.")
            return None
    elif c is None:
        if (np.sqrt(len(vec)).is_integer()):
            # matrix is square
            c = int(np.sqrt(len(vec)))
        else:
            print("Error: vector cannot form a square matrix. \
                  Please provide a column length, c.")
            return None
    elif (not (len(vec) / c).is_integer()):
        # c does not divide length of vec
        print("Error: value of c is invalid. \
              Cannot split vector evenly into columns of length c")
        return None

    # number of rows
    n = int(len(vec) / c)

    return vec.reshape((c, n), order='F')


def is_square(arr):
    """Return if array is square or not.
    """
    try:
        return arr.shape[0] == arr.shape[1]
    except:
        return False


def liouvillian(H):
    """
    Return Liouvillian of a Hamiltonian.

    Parameters
    ----------
    H : ndarray
        Square matrix with dimension n.

    Returns
    -------
    ndarray
        Square matrix with dimension n^2.

    """
    n = H.shape[0]

    return -1j * (np.kron(eye(n), H) - np.kron(H.T, eye(n)))


def commutator(A, B, kind="normal"):
    """
    Return commutator of kind of A and B.

    Parameters
    ----------
    A : ndarray
        Square array.
    B : ndarray
        Square array.
    kind : str, optional
        kind of commutator (normal, anti), The default is "normal".

    Returns
    -------
    ndarray
        Commutator of A and B.

    """
    if kind == "normal":
        return A@B - B@A
    elif kind == "anti":
        return A@B + B@A
    else:
        raise TypeError("Unknown commutator kind " + str(kind))


def kron(*args):
    """
    Return Kronecker product of input arguments.

    Returns
    -------
    ndarray
        Kronecker product.

    Raises
    ------
    TypeError
        No input arguments.

    """
    if not args:
        raise TypeError("Requires at least one input argument")

    if len(args) == 1 and isinstance(args[0], list):
        # input of the form [a,b,...]
        mlist = args[0]
    elif len(args) == 1 and isinstance(args[0], np.ndarray):
        # single
        if len(args[0].shape) == 2:
            return args[0]
        else:
            # ndarray
            mlist = args[0]
    else:
        mlist = args

    out = mlist[0]
    for m in mlist[1:]:
        out = np.kron(out, m)

    return out


def timegrid(start, stop, step, dtype=None):
    """
    Return numbers spaced by specified step over specified interval.

    Parameters
    ----------
    start : array_like
        Starting value of sequence.
    stop : array_like
        End value of sequence.
    step : array_like
        Amount by which to space points in sequence.
    dtype : dtype, optional
        The type of the output array. If dtype is not given,then the data type 
        is inferred from arguments. The default is None.

    Returns
    -------
    ndarray
        Equally spaced numbers as specified.

    """

    return (np.linspace(start, stop, int((stop - start) / step) + 1)
            .astype(dtype))


def frobenius(a, b):
    """
    Return Frobenius/trace inner product of a and b.
    Applied element-wise if a is not single.

    Parameters
    ----------
    a : ndarray
        Square array or list/array of square arrays.
    b : ndarray
        Square array.

    Returns
    -------
    ndarray or scalar
        The value(s) of the trace of a times b.

    Examples
    --------
    >>> mp.Frobenius(mp.eye(2), np.ones((2,2)))
    (2+0j)
    >>> a = [mp.eye(2), np.ones((2,2))]
    >>> mp.Frobenius(a, np.ones((2,2)))
    array([2.+0.j, 4.+0.j])

    """
    a = np.asarray(a, dtype=object)
    b = np.asarray(b, dtype=complex)

    try:
        # a is an array
        t = []
        for x in a:
            t.append(np.trace(x.conj().T @ b))
        return np.asarray(t)

    except:
        # a is single
        return np.trace(a.conj().T @ b)