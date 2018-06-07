from src import plotter
import argparse


if __name__ == "__main__":
    # argparse setup
    parser = argparse.ArgumentParser(
        description='Plotter'
    )

    parser.add_argument(
        '-p',
        '--plots',
        nargs='+',
        help='List of plots to create',
        required=False,
        action='store',
        dest='plots',
        default=()
    )

    parser.add_argument(
        '-t', '--test_path',
        action='store',
        default=None,
        required=False,
        help='Test from which load logs to',
        dest='test_path'
    )

    parser.add_argument(
        '-s', '--save',
        action='store_true',
        default=False,
        required=False,
        help='Specify whether save to file or not',
        dest='s_flag'
    )

    args = parser.parse_args()

    plotter.plot_from_files(
        test_folder_path=args.test_path,
        save_plots_to_test_folder=args.s_flag,
        instant_plot=not args.s_flag,
        plots=tuple(args.plots)
    )