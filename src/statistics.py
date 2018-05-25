import math, sys, random, abc


def single_iteration_velocity_lower_bound(degree, distr_class, *distr_param):
    try :
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
        return random.expovariate(lambd)

    @staticmethod
    def mean(lambd):
        return 1 / lambd

    @staticmethod
    def variance(lambd):
        return 1 / (lambd ** 2)


class UniformDistribution(DistributionAbstract):
    @staticmethod
    def sample(*args):
        a, b = UniformDistribution.get_interval_extremes(*args)
        return random.uniform(a, b)

    @staticmethod
    def get_interval_extremes(*args):
        a = b = 0
        if len(args) == 1:
            b = args[0]
        elif len(args) == 2:
            a = args[0]
            b = args[1]
        else:
            raise Exception("Unexpected params args for uniform distribution")
        return a, b

    @staticmethod
    def mean(*args):
        a, b = UniformDistribution.get_interval_extremes(*args)
        return (a + b) / 2

    @staticmethod
    def variance(*args):
        a, b = UniformDistribution.get_interval_extremes(*args)
        return 1 / 12 * (a - b) ** 2


class ParetoTimeDistribution(DistributionAbstract):
    @staticmethod
    def sample(alpha):
        return random.paretovariate(alpha)

    @staticmethod
    def mean(alpha):
        if alpha <= 1:
            return math.inf
        return alpha / (alpha - 1)

    @staticmethod
    def variance(alpha):
        if alpha <= 2:
            return math.inf
        return alpha / ((alpha - 1) ** 2 * (alpha - 2) ** 2)


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
    def sample(*args, k=1):
        raise NotImplementedError("sample method is not defined for this distribution")

    @staticmethod
    def mean(*args, k=1):
        a, b = UniformDistribution.get_interval_extremes(*args)
        return (a + b * k) / (k + 1)

    @staticmethod
    def variance(*args, k=1):
        raise NotImplementedError("variance method is not defined for this distribution")


class MaxOfParetoDistribution(DistributionAbstract):
    @staticmethod
    def sample(alpha, k=1):
        raise NotImplementedError("variance method is not defined for this distribution")

    @staticmethod
    def mean(alpha, k=1):
        raise NotImplementedError("variance method is not defined for this distribution")

    @staticmethod
    def variance(alpha, k=1):
        raise NotImplementedError("variance method is not defined for this distribution")
