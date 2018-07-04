import math, sys, random, abc, decimal, os
from scipy import integrate
import numpy as np

D = decimal.Decimal
decimal.getcontext().prec = 256

def single_iteration_velocity_as_tot_iters_over_avg_diagonal_iter(test_folder_path, graph, nodes_amount, max_time):
    graph_avg_iter_log_path = "{}/{}_avg_iter_time_log".format(test_folder_path, graph)
    diag_avg_iter_log_path = "{}/0_diagonal_avg_iter_time_log".format(test_folder_path)

    if os.path.isfile(graph_avg_iter_log_path + ".gz"):
        graph_avg_iter_log_path += '.gz'

    if os.path.isfile(diag_avg_iter_log_path + ".gz"):
        diag_avg_iter_log_path += '.gz'

    graph_avg_iter_log = [tuple(s.split(",")) for s in np.loadtxt(graph_avg_iter_log_path, str)]
    diag_avg_iter_log = [tuple(s.split(",")) for s in np.loadtxt(diag_avg_iter_log_path, str)]

    diag_avg_iter_time = float(diag_avg_iter_log[-1][0])
    diag_avg_iter_iter = float(diag_avg_iter_log[-1][1])
    graph_avg_iter_time = float(graph_avg_iter_log[-1][0])
    graph_avg_iter_iter = float(graph_avg_iter_log[-1][1])

    if 0 != abs(diag_avg_iter_iter - graph_avg_iter_iter) <= nodes_amount:
        raise Exception("You're using logs generated from a simulation using max_iter limit, this kind of simulation "
                        "doesn't allow to compute values for this function")

    return graph_avg_iter_iter / max_time


def single_iteration_velocity_residual_lifetime_lower_bound(degree, distr_class, distr_params):
    if not type(distr_params) in (list, tuple,):
        distr_params = (distr_params,)
    E_X = distr_class.mean(*distr_params)
    E_Z = eval("MaxOf" + distr_class.__name__).residual_time_mean(*distr_params, k=degree)
    return 1 / (E_X + E_Z)


def single_iteration_velocity_memoryless_lower_bound(degree, distr_class, distr_params):
    if not type(distr_params) in (list, tuple,):
        distr_params = (distr_params,)
    E_X = distr_class.mean(*distr_params)
    E_Z = eval("MaxOf" + distr_class.__name__).mean(*distr_params, k=degree)
    return 1 / (E_X + E_Z)


def single_iteration_velocity_upper_bound(degree, distr_class, distr_params):
    if not type(distr_params) in (list, tuple,):
        distr_params = (distr_params,)
    E_X = distr_class.mean(*distr_params)
    E_Z = eval("MaxOf" + distr_class.__name__).mean(*distr_params, k=(degree + 1))
    return 2 / (E_X + E_Z)


def single_iteration_velocity_don_bound(degree, distr_class, distr_params, out_decimal=False):
    if not type(distr_params) in (list, tuple,):
        distr_params = (distr_params,)
    k = D(degree)
    s = D(0.0)
    for h in range(1, degree + 1):
        E_max_Xk = eval("MaxOf" + distr_class.__name__).residual_time_mean(*distr_params, k=h, out_decimal=True)
        s += D(binomial(k, h)) * D(1 / 2 ** k) * D(E_max_Xk)

    E_X = distr_class.mean(*distr_params, out_decimal=True)
    velocity = D(1 / (s + E_X))

    if out_decimal:
        return velocity
    return float(velocity)


class DistributionAbstract:
    __metaclass__ = abc.ABCMeta

    @staticmethod
    @abc.abstractmethod
    def sample(param):
        raise NotImplementedError('sample method not implemented in TimeDistributionAbstract child class')

    @staticmethod
    @abc.abstractmethod
    def mean(param):
        raise NotImplementedError('mean method not implemented in TimeDistributionAbstract child class')

    @staticmethod
    @abc.abstractmethod
    def variance(param):
        raise NotImplementedError('variance method not implemented in TimeDistributionAbstract child class')


class ExponentialDistribution(DistributionAbstract):
    name = "Exponential"
    shortname = 'exp'

    @staticmethod
    def sample(lambd, out_decimal=False):
        if out_decimal:
            return D(np.random.exponential(1 / lambd))
        return np.random.exponential(1 / lambd)

    @staticmethod
    def mean(lambd, out_decimal=False):
        if out_decimal:
            return D(1 / lambd)
        return 1 / lambd

    @staticmethod
    def variance(lambd, out_decimal=False):
        if out_decimal:
            return D(1 / lambd ** 2)
        return 1 / lambd ** 2


class UniformDistribution(DistributionAbstract):
    name = "Uniform"
    shortname = 'unif'

    @staticmethod
    def F_X(x, a=0, b=1):
        if x <= a:
            return 0
        elif a < x < b:
            return (x - a) / (b - a)
        else:
            return 1

    @staticmethod
    def F_res_X(x, a=0, b=1, out_decimal=False):
        if out_decimal:
            return D((a ** 2 - 2 * b * x + x ** 2) / (a ** 2 - b ** 2))
        return (a ** 2 - 2 * b * x + x ** 2) / (a ** 2 - b ** 2)

    @staticmethod
    def sample(a=0, b=1):
        return np.random.uniform(a, b)

    @staticmethod
    def mean(a=0, b=1, out_decimal=False):
        if out_decimal:
            return D((a + b) / 2)
        return (a + b) / 2

    @staticmethod
    def variance(a=0, b=1):
        return 1 / 12 * (a - b) ** 2


class Type2ParetoDistribution(DistributionAbstract):
    name = "Type II Pareto"
    shortname = 'lomax'

    @staticmethod
    def sample(alpha, sigma=1):
        return np.random.pareto(alpha) * sigma

    @staticmethod
    def mean(alpha, sigma=1, out_decimal=False):
        if alpha <= 1:
            return math.inf
        if out_decimal:
            return D(sigma / (alpha - 1))

        return sigma / (alpha - 1)

    @staticmethod
    def variance(alpha, sigma=1):
        if alpha <= 2:
            return math.inf
        return (alpha * sigma ** 2) / ((alpha - 1) ** 2 * (alpha - 2))


class MaxOfExponentialDistribution(DistributionAbstract):
    @staticmethod
    def sample(lambd, k=1, out_decimal=False):
        raise NotImplementedError("sample method is not defined for MaxOfExponentialDistribution distribution")

    @staticmethod
    def mean(lambd, k=1, out_decimal=False):
        if out_decimal:
            return D(1 / D(lambd) * (sum(D(1 / i) for i in range(1, k + 1))))

        return 1 / lambd * (sum((1 / i) for i in range(1, k + 1)))

    @staticmethod
    def residual_time_mean(lambd, k=1, out_decimal=False):
        return MaxOfExponentialDistribution.mean(lambd, k, out_decimal)

    @staticmethod
    def variance(lambd, k=1):
        raise NotImplementedError("variance method is not defined for MaxOfExponentialDistribution distribution")


class MaxOfUniformDistribution(DistributionAbstract):
    @staticmethod
    def sample(a=0, b=1, k=1):
        raise NotImplementedError("sample method is not defined for MaxOfUniformDistribution distribution")

    @staticmethod
    def mean(a=0, b=1, k=1, out_decimal=False):
        if out_decimal:
            return D((a + b * k) / (k + 1))
        return (a + b * k) / (k + 1)

    @staticmethod
    def residual_time_mean(a=0, b=1, k=1, out_decimal=False):
        integrand = lambda y: (y ** 2 - 2 * b * y + a ** 2) ** k
        integral = integrate.quad(integrand, a, b)
        # integrand = lambda x: 1 - UniformDistribution.F_res_X(x, *args) ** k
        # print(integral[1])
        res_time_mean = b - D((2 ** k * a ** (k + 1)) / ((a + b) ** k * (k + 1))) - D(1 / (a ** 2 - b ** 2) ** k) * \
                        D(integral[0])

        if out_decimal:
            return res_time_mean

        return float(res_time_mean)

    @staticmethod
    def variance(a=0, b=1, k=1):
        raise NotImplementedError("variance method is not defined for MaxOfUniformDistribution distribution")


class MaxOfType2ParetoDistribution(DistributionAbstract):
    @staticmethod
    def sample(alpha, sigma=2, k=1):
        raise NotImplementedError("sample method is not defined for MaxOfType2ParetoDistribution distribution")

    @staticmethod
    def mean(alpha, sigma=2, k=1, out_decimal=False):
        a = D(alpha)
        s = D(sigma)
        summation = D(0.0)
        for i in range(1, k + 1):
            d1 = D(binomial(k, i))
            d2 = D(-1.0) ** (D(i) + D(1.0))
            d3 = D(s) / (D(a) * D(i) - D(1.0))
            summation += d1 * d2 * d3

        if out_decimal:
            return summation

        return float(summation)

    @staticmethod
    def residual_time_mean(alpha, sigma=2, k=1, out_decimal=False):
        a = D(alpha)
        s = D(sigma)
        summation = D(0.0)
        for i in range(1, k + 1):
            """
            if not True:
                integrand = lambda y: (1 + (y / s)) ** ((-a + 1) * i)
                intgr = integrate.quad(integrand, 0, np.inf)
                # print(intgr[1])
                intgr = intgr[0]
            else:
            """

            d1 = D(binomial(k, i))
            d2 = D(-1.0) ** (D(i) + D(1.0))
            d3 = D(s) / ((D(a) - D(1.0)) * D(i) - D(1.0))
            summation += d1 * d2 * d3

        if out_decimal:
            return summation

        return float(summation)

    @staticmethod
    def variance(alpha, sigma=2, k=1):
        raise NotImplementedError("variance method is not defined for MaxOfType2ParetoDistribution distribution")


def binomial(n, k):
    if n == k:
        return 1
    elif k == 1:
        return n
    elif k > n:
        return 0
    else:
        a = math.factorial(n)
        b = math.factorial(k)
        c = math.factorial(n - k)
        return a // (b * c)
