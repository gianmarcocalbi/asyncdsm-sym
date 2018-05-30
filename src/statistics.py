import math, sys, random, abc, decimal
from scipy import integrate
import numpy as np
import sympy as sp


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
    E_Z = eval("MaxOf" + distr_class.__name__).mean(*distr_params, k=degree)
    return 2 / (E_X + E_Z)

def single_iteration_velocity_don_bound(degree, distr_class, distr_params):
    if not distr_class is MaxOfExponentialDistribution:
        raise Exception("This bound is not defined for MaxOf{} class".format(distr_class.__name__))

    if not type(distr_params) in (list, tuple,):
        distr_params = (distr_params,)
    E_X = distr_class.mean(*distr_params)
    E_Z = eval("MaxOf" + distr_class.__name__).mean(*distr_params, k=degree)
    return 2 / (E_X + E_Z)

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
    @staticmethod
    def sample(lambd):
        return np.random.exponential(1 / lambd)

    @staticmethod
    def mean(lambd):
        return 1 / lambd

    @staticmethod
    def variance(lambd):
        return 1 / (lambd ** 2)


class UniformDistribution(DistributionAbstract):
    @staticmethod
    def F_X(x, a=0, b=1):
        if x <= a:
            return 0
        elif a < x < b:
            return (x - a) / (b - a)
        else:
            return 1

    @staticmethod
    def F_res_X(x, a=0, b=1):
        return (a ** 2 - 2 * b * x + x ** 2) / (a ** 2 - b ** 2)

    @staticmethod
    def sample(a=0, b=1):
        return np.random.uniform(a, b)

    @staticmethod
    def mean(a=0, b=1):
        return (a + b) / 2

    @staticmethod
    def variance(a=0, b=1):
        return 1 / 12 * (a - b) ** 2


class Type2ParetoDistribution(DistributionAbstract):
    @staticmethod
    def sample(alpha, sigma=1):
        return np.random.pareto(alpha) * sigma

    @staticmethod
    def mean(alpha, sigma=1):
        if alpha <= 1:
            return math.inf
        return sigma / (alpha - 1)

    @staticmethod
    def variance(alpha, sigma=1):
        if alpha <= 2:
            return math.inf
        return (alpha * sigma ** 2) / ((alpha - 1) ** 2 * (alpha - 2))


class MaxOfExponentialDistribution(DistributionAbstract):
    @staticmethod
    def sample(lambd, k=1):
        raise NotImplementedError("sample method is not defined for MaxOfExponentialDistribution distribution")

    @staticmethod
    def mean(lambd, k=1):
        return 1 / lambd * (sum((1 / i) for i in range(1, k + 1)))

    @staticmethod
    def residual_time_mean(lambd, k=1):
        return MaxOfExponentialDistribution.mean(lambd, k)

    @staticmethod
    def variance(lambd, k=1):
        raise NotImplementedError("variance method is not defined for MaxOfExponentialDistribution distribution")


class MaxOfUniformDistribution(DistributionAbstract):
    @staticmethod
    def sample(a=0, b=1, k=1):
        raise NotImplementedError("sample method is not defined for MaxOfUniformDistribution distribution")

    @staticmethod
    def mean(a=0, b=1, k=1):
        return (a + b * k) / (k + 1)

    @staticmethod
    def residual_time_mean(a=0, b=1, k=1):
        integrand = lambda y: (y ** 2 - 2 * b * y + a ** 2) ** k
        integral = integrate.quad(integrand, a, b)
        # integrand = lambda x: 1 - UniformDistribution.F_res_X(x, *args) ** k
        # print(integral[1])
        return b - (2 ** k * a ** (k + 1)) / ((a + b) ** k * (k + 1)) - (1 / (a ** 2 - b ** 2) ** k) * integral[0]

    @staticmethod
    def variance(a=0, b=1, k=1):
        raise NotImplementedError("variance method is not defined for MaxOfUniformDistribution distribution")


class MaxOfType2ParetoDistribution(DistributionAbstract):
    @staticmethod
    def sample(alpha, sigma=2, k=1):
        raise NotImplementedError("sample method is not defined for MaxOfType2ParetoDistribution distribution")

    @staticmethod
    def mean(alpha, sigma=2, k=1):
        prev_prec = decimal.getcontext().prec
        D = decimal.Decimal
        decimal.getcontext().prec = 128

        a = D(alpha)
        s = D(sigma)
        summation = D(0.0)
        for i in range(1, k + 1):
            d1 = D(binomial(k, i))
            d2 = D(-1.0) ** (D(i) + D(1.0))
            d3 = D(s) / (D(a) * D(i) - D(1.0))
            summation += d1 * d2 * d3

        decimal.getcontext().prec = prev_prec

        return float(summation)

    @staticmethod
    def residual_time_mean(alpha, sigma=2, k=1):
        prev_prec = decimal.getcontext().prec
        D = decimal.Decimal
        decimal.getcontext().prec = 128

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

        decimal.getcontext().prec = prev_prec

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
