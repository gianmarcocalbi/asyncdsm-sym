from plotter_new import Plotter
from src import statistics

p = Plotter(test_folder_name='test_006_exp1lambda_1ktime1e-4alphaXin0-2_classic.conflict.1')
p.plot_avg_iter_over_time_with_don_bound()
p.plot_don_bound_error_over_degree()