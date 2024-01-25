import gpflow
from check_shapes import check_shapes, inherit_check_shapes
from gpflow.base import TensorType
from gpflow.config import default_float
import tensorflow as tf
import numpy as np


class LinearNoise(gpflow.functions.Function):
    """
    This class represents a linear noise function of the form y_i = c + A * x0_i.

    Parameters:
    - c (TensorType): Additive constant (default is None).
    - A (TensorType): Matrix mapping each element of X to Y (default is None).

    Attributes:
    - A (gpflow.Parameter): Matrix parameter representing the mapping.
    - c (gpflow.Parameter): Parameter representing the additive constant.

    Methods:
    - __init__(self, c: TensorType = None, A: TensorType = None) -> None
    - __call__(self, X: TensorType) -> tf.Tensor
    """

    @check_shapes(
        "c: [broadcast Q]",
        "A: [broadcast D, broadcast Q]",
    )
    def __init__(self, c: TensorType = None, A: TensorType = None) -> None:
        """
        Initializes the LinearNoise function with provided parameters.

        Parameters:
        - c (TensorType): Additive constant (default is None).
        - A (TensorType): Matrix mapping each element of X to Y (default is None).

        Returns:
        None
        """
        A = np.ones((1, 1), dtype=default_float()) if A is None else A
        c = np.zeros(1, dtype=default_float()) if c is None else c
        self.A = gpflow.Parameter(np.atleast_2d(A))
        self.c = gpflow.Parameter(c)

    @inherit_check_shapes
    def __call__(self, X: TensorType) -> tf.Tensor:
        """
        Computes the output of the linear noise function for given input X.

        Parameters:
        - X (TensorType): Input tensor.

        Returns:
        tf.Tensor: Output tensor of the linear noise function.
        """
        X0 = X[:, 0]
        X0 = tf.reshape(X0, (-1, 1))
        X1 = X[:, 1]
        X1 = tf.reshape(X1, (-1, 1))

        return tf.tensordot(X0, self.A, [[-1], [0]]) + self.c
    

# Mean function that uses a weighting parameter alpha and combines to modulation by a gamma and a temperature function
class ViewpixxAlpha(gpflow.functions.MeanFunction, gpflow.functions.Function):
    """
    This class represents a mean function that combines modulation by a gamma and a temperature function
    using a weighting parameter alpha.

    The function is defined as:
    y_i = alpha * (c1 + A * x0_i ** gamma) + (1-alpha) * (c2 + C * x1_i **2 + D * x0_i * x1_i - D * x0_i**2 * x1_i)

    Parameters:
    - gamma (TensorType): Exponent parameter for the modulation (default is 2.2).
    - alpha (TensorType): Weighting parameter (default is 0).
    - c1 (TensorType): Additive constant for the first part (default is 0).
    - c2 (TensorType): Additive constant for the second part (default is 0).
    - A (TensorType): Matrix mapping each element of X0 to the modulated output (default is 1).
    - C (TensorType): Matrix mapping each element of X1 to the temperature-modulated output (default is 1).
    - D (TensorType): Matrix mapping each element of X0 and X1 to the temperature-modulated output (default is 1).

    Attributes:
    - gamma (gpflow.Parameter): Exponent parameter for the modulation.
    - alpha (gpflow.Parameter): Weighting parameter.
    - c1 (gpflow.Parameter): Additive constant for the first part.
    - c2 (gpflow.Parameter): Additive constant for the second part.
    - A (gpflow.Parameter): Matrix parameter for the mapping of X0.
    - C (gpflow.Parameter): Matrix parameter for the mapping of X1 for temperature modulation.
    - D (gpflow.Parameter): Matrix parameter for the mapping of X0 and X1 for temperature modulation.

    Methods:
    - __init__(self, gamma: TensorType = 2.2, alpha: TensorType = 0, c1: TensorType = 0, c2: TensorType = 0, A: TensorType = 1, C: TensorType = 1, D: TensorType = 1) -> None
    - __call__(self, X: TensorType) -> tf.Tensor
    """

    @check_shapes(
        "gamma: [broadcast Q]",
        "alpha: [broadcast Q]",
        "c1: [broadcast Q]",
        "c2: [broadcast Q]",
        "A: [broadcast D, broadcast Q]",
        "C: [broadcast D, broadcast Q]",
        "D: [broadcast D, broadcast Q]",
    )
    def __init__(self, gamma: TensorType = None, alpha: TensorType = None, c1: TensorType = None, c2: TensorType = None, A: TensorType = None, C: TensorType = None, D: TensorType = None) -> None:
        """
        Initializes the ViewpixxAlpha mean function with provided parameters.

        Parameters:
        - gamma (TensorType): Exponent parameter for the modulation (default is 2.2).
        - alpha (TensorType): Weighting parameter (default is 0).
        - c1 (TensorType): Additive constant for the first part (default is 0).
        - c2 (TensorType): Additive constant for the second part (default is 0).
        - A (TensorType): Matrix mapping each element of X0 to the modulated output (default is 1).
        - C (TensorType): Matrix mapping each element of X1 to the temperature-modulated output (default is 1).
        - D (TensorType): Matrix parameter for the mapping of X0 and X1 for temperature modulation (default is 1).

        Returns:
        None
        """
        gpflow.functions.MeanFunction.__init__(self)
        gamma = 2.2 if gamma is None else gamma
        alpha = np.zeros(1, dtype=default_float()) if alpha is None else alpha
        c1 = np.zeros(1, dtype=default_float()) if c1 is None else c1
        c2 = np.zeros(1, dtype=default_float()) if c2 is None else c2
        A = np.ones((1, 1), dtype=default_float()) if A is None else A
        C = np.ones((1, 1), dtype=default_float()) if C is None else C
        D = np.ones((1, 1), dtype=default_float()) if D is None else D
        self.gamma = gpflow.Parameter(gamma)
        self.alpha = gpflow.Parameter(alpha)
        self.c1 = gpflow.Parameter(c1)
        self.c2 = gpflow.Parameter(c2)
        self.A = gpflow.Parameter(np.atleast_2d(A))
        self.C = gpflow.Parameter(np.atleast_2d(C))
        self.D = gpflow.Parameter(np.atleast_2d(D))

    @inherit_check_shapes
    def __call__(self, X: TensorType) -> tf.Tensor:
        """
        Computes the output of the ViewpixxAlpha mean function for the given input X.

        Parameters:
        - X (TensorType): Input tensor.

        Returns:
        tf.Tensor: Output tensor of the ViewpixxAlpha mean function.
        """
        X0 = X[:, 0]
        X0 = tf.reshape(X0, (-1, 1))
        X1 = X[:, 1]
        X1 = tf.reshape(X1, (-1, 1))

        partL_in = self.c1 + tf.tensordot(X0**self.gamma, self.A, [[-1], [0]])
        partTemp = self.c2 + tf.tensordot(X1**2, self.C, [[-1], [0]]) + tf.tensordot(X0*X1, self.D, [[-1], [0]]) + tf.tensordot(X0**2*X1, -self.D, [[-1], [0]])
        return self.alpha * partL_in + (1-self.alpha) * partTemp