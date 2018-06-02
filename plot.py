from src.plotter import Plotter

p = Plotter(test_folder_name='test_006_exp1lambda_1ktime1e-4alphaXin0-2_classic.conflict.1')
p.plot_iter_all_bounds_error_over_degree_with_real_velocity()
p.plot_iter_all_bounds_velocity_over_degrees_with_real_velocity()