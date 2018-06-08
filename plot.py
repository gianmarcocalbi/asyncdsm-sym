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
        '-f', '--folder-path',
        action='store',
        default=None,
        required=False,
        help='Test folder from which load logs to',
        dest='folder_path'
    )

    parser.add_argument(
        '-t', '--temp-index',
        action='store',
        default=0,
        required=False,
        help='Test folder from which load logs to',
        dest='temp_index'
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
        test_folder_path=args.folder_path,
        save_plots_to_test_folder=args.s_flag,
        instant_plot=not args.s_flag,
        plots=tuple(args.plots),
        temp_index=int(args.temp_index)
    )