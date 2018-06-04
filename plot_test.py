from src.plotter import Plotter

Plotter(
    test_folder_path='./test_log/test_006_exp1lambda_10ktime1e-4alphaXin0-2_classic',
    save_plots_to_test_folder=False,
    instant_plot=True,
    plots=(
        "iter_time",
        "avg-iter_time",
        #"avg-iter_time-memoryless-lb",
        #"avg-iter_time-residual-lifetime-lb",
        #"avg-iter_time-ub",
        #"avg-iter_time-don-bound",
        "mse_iter",
        "real-mse_iter",
        "mse_time",
        "real-mse_time",
        #"iter-memoryless-lb-error_degree",
        #"iter-residual-lifetime-lb-error_degree",
        #"iter-ub-error_degree",
        "iter-all-bounds-error_degree",
        #"iter-memoryless-lb-velocity_degree",
        #"iter-residual-lifetime-lb-velocity_degree",
        #"iter-ub-velocity_degree",
        "iter-all-bounds-velocity_degree",
    ),
    moving_average_window=0,
    ymax=None,
    yscale='log',  # linear or log
    scatter=False,
    points_size=0.5
).plot()