from src.plotter import Plotter
from matplotlib import pyplot as plt
from src.utils import *

logs1, setup1 = load_test_logs(
    './test_log/test_cyc1_eigvecsvm_C0.1alpha_100n_noshuf_exp[1]_mtrT2worst_INFtime_1000iter'
)
logs2, setup2 = load_test_logs(
    './test_log/test_cyc2_eigvecsvm_C0.1alpha_100n_noshuf_exp[1]_mtrT2worst_INFtime_1000iter'
)

# plt.title('Min iteration VS time', loc='left')
plt.xlabel('Time')
plt.ylabel('Iteration')

xlim = 0

y1 = logs1['metrics']['hinge_loss']['2-undir_cycle']
y2 = logs2['metrics']['cont_hinge_loss']['2-undir_cycle']
x = list(range(1001))


plt.plot(
    x,
    y1,
    label='HL',
    color='b'
)
plt.plot(
    x,
    y2,
    label='cHL',
    color='g'
)


plt.yscale('linear')
plt.legend()
plt.show()
plt.close()
