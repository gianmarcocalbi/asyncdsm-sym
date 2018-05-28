import math, sys, random, abc, decimal
from scipy import integrate
from scipy.special import binom
import numpy as np


def single_iteration_velocity_residual_lifetime_lower_bound(degree, distr_class, distr_param):
    if not type(distr_param) in (list, tuple,):
        distr_param = (distr_param,)
    E_X = distr_class.mean(*distr_param)
    E_Z = eval("MaxOf" + distr_class.__name__).residual_time_mean(*distr_param, k=degree)
    return 1 / (E_X + E_Z)


def single_iteration_velocity_memoryless_lower_bound(degree, distr_class, *distr_param):
    try:
        E_X = distr_class.mean(distr_param)
        new_distr_param = distr_param
    except TypeError:
        E_X = distr_class.mean(distr_param[0])
        new_distr_param = distr_param[0]

    E_Z = eval("MaxOf" + distr_class.__name__).mean(new_distr_param, k=degree)
    return 1 / (E_X + E_Z)


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
        raise NotImplementedError("sample method is not defined for this distribution")

    @staticmethod
    def mean(lambd, k=1):
        return 1 / lambd * (sum((1 / i) for i in range(1, k + 1)))

    @staticmethod
    def variance(lambd, k=1):
        raise NotImplementedError("variance method is not defined for this distribution")


class MaxOfUniformDistribution(DistributionAbstract):
    @staticmethod
    def sample(a=0, b=1, k=1):
        raise NotImplementedError("sample method is not defined for this distribution")

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
        raise NotImplementedError("variance method is not defined for this distribution")


class MaxOfType2ParetoDistribution(DistributionAbstract):
    @staticmethod
    def sample(alpha, sigma=2, k=1):
        raise NotImplementedError("variance method is not defined for this distribution")

    @staticmethod
    def mean(alpha, sigma=2, k=1):
        raise NotImplementedError("variance method is not defined for this distribution")

    @staticmethod
    def residual_time_mean(alpha, sigma=2, k=1):
        prev_prec = decimal.getcontext().prec
        D = decimal.Decimal
        decimal.getcontext().prec = 64

        a = float(alpha)
        s = float(sigma)
        _sumD = D(0.0)
        _sum1 = 0.0
        _sum2 = 0.0
        _sum3 = 0.0
        for i in range(1, k + 1):
            """
            if not True:
                integrand = lambda y: (1 + (y / s)) ** ((-a + 1) * i)
                intgr = integrate.quad(integrand, 0, np.inf)
                # print(intgr[1])
                intgr = intgr[0]
            else:
            """

            p1 = float(binom(k, i))
            p2 = float((-1) ** (i + 1))
            p3 = s / ((a - 1) * i - 1)
            _sum1 += p1 * p2 * p3

            intgr = (D(s) / ((D(a) - D(1.0)) * D(i) - D(1.0)))
            d1 = D(binom(k, i))
            d2 = D(-1.0) ** (D(i) + D(1.0))
            d3 = D(intgr)
            _sumD += d1 * d2 * d3

            _sum2 += (((-1) ** (i + 1)) * binom(k, i)) * (s / ((a-1) * i - 1.0))
            _sum3 += (((-1) ** (i + 1)) * binom(k, i)) * s / ((a-1) * i - 1.0)

        decimal.getcontext().prec = prev_prec
        #return _sum1, _sum2, _sum3, _sumD
        return float(_sumD)

    @staticmethod
    def variance(alpha, sigma=2, k=1):
        raise NotImplementedError("variance method is not defined for this distribution")
