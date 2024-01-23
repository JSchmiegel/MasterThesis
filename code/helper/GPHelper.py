import gpflow
from check_shapes import check_shapes, inherit_check_shapes
from gpflow.base import TensorType
from gpflow.config import default_float
import tensorflow as tf
import numpy as np


# Function to use as linear Noise
class LinearNoise(gpflow.functions.Function):
    """
    y_i = c + A * x0_i 
    """

    @check_shapes(
        "c: [broadcast Q]",
        "A: [broadcast D, broadcast Q]",
    )
    def __init__(self, c: TensorType = None, A: TensorType = None) -> None:
        """
        A is a matrix which maps each element of X to Y, b is an additive
        constant.
        """
        A = np.ones((1, 1), dtype=default_float()) if A is None else A
        c = np.zeros(1, dtype=default_float()) if c is None else c
        self.A = gpflow.Parameter(np.atleast_2d(A))
        self.c = gpflow.Parameter(c)

    @inherit_check_shapes
    def __call__(self, X: TensorType) -> tf.Tensor:
        X0 = X[:, 0]
        X0 = tf.reshape(X0, (-1, 1))
        X1 = X[:, 1]
        X1 = tf.reshape(X1, (-1, 1))

        return tf.tensordot(X0, self.A, [[-1], [0]]) + self.c
    

# Mean function that uses a weighting parameter alpha and combines to modulation by a gamma and a temperature function
class ViewpixxAlpha(gpflow.functions.MeanFunction, gpflow.functions.Function):
    """
    y_i = alpha *   (c1 + A * x0_i ** gamma) + 
          (1-alpha) (c2 + C * x1_i **2 + D * x0_i * x1_i + -D * x0_i**2 * x1_i)

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
        X0 = X[:, 0]
        X0 = tf.reshape(X0, (-1, 1))
        X1 = X[:, 1]
        X1 = tf.reshape(X1, (-1, 1))

        partL_in = self.c1 + tf.tensordot(X0**self.gamma, self.A, [[-1], [0]])
        partTemp = self.c2 + tf.tensordot(X1**2, self.C, [[-1], [0]]) + tf.tensordot(X0*X1, self.D, [[-1], [0]]) + tf.tensordot(X0**2*X1, -self.D, [[-1], [0]])
        return self.alpha * partL_in + (1-self.alpha) * partTemp