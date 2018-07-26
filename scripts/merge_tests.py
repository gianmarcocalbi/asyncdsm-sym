import glob, os, pickle, datetime, time, re, pprint
import matplotlib.pyplot as plt
import numpy as np
from src import plotter, graphs
from src.mltoolbox.metrics import METRICS
from src.utils import *
from shutil import copyfile, rmtree


def main():
    # SETUP BEGIN

    """
    x01 : reg only uniform edges avg on 8 tests
    x02 : reg cycles avg on 8 tests
    x03 : reg all with avg on 16 tests
    x04 : svm
    """

    test_suite_root = './test_log/bulk/'
    test_suite_code = 'r2001'
    test_suite_pattern = 'test_r2001_reg2*'
    log_pattern = re.compile(r'.*')  # re.compile(r'.*mse_log\.gz$')
    excluded_graphs = []

    # SETUP END

    # list paths of all folders of tests that satisfy the setting pattern
    test_folder_paths = [s.replace('\\', '/') for s in list(glob.iglob(
        os.path.normpath("{}{}/{}".format(
            test_suite_root,
            test_suite_code,
            test_suite_pattern
        ).replace('\\', '/'))))
    ]

    # if paths list if empty abort with error
    if len(test_folder_paths) == 0:
        raise Exception("Empty test folder paths list")

    # setup file and metrics (are initially taken from the first simulation in the path list)
    setup = None
    setup_metrics = set([])
    setup_real_metrics = set([])
    """
    logs = {
        $TEST_FOLDER_PATH : {
            $TEST_LOG_FILENAME : $LOG_ARRAY
        }
    }
    """
    logs = {}

    for test_folder_path in test_folder_paths[:]:
        if re.match(r'.*.AVG$', test_folder_path):
            # if the folder is result of a previous merge then skip it
            test_folder_paths.remove(test_folder_path)
            continue
        try:
            # open current test setup file
            with open("{}/.setup.pkl".format(test_folder_path), 'rb') as setup_file:
                _setup = pickle.load(setup_file)
        except:
            print("No setup file to open")
            raise


        if setup is None:
            # is setup variable is not set then set setup and setup metrics equal to those
            # of the current opened test folder
            setup = _setup
            setup_metrics = set(setup['metrics'])
            setup_real_metrics = set(setup['real_metrics'])
        else:
            # is setup is already set then intersect metrics and real metrics to keep only
            # those shared among all test folders
            setup_metrics &= set(setup['metrics'])
            setup_real_metrics &= set(setup['real_metrics'])

        logs[test_folder_path] = {}

        # list all logs inside current test folder escaping problematic characters
        test_logs_paths = [s.replace('\\', '/') for s in list(
            glob.iglob(
                os.path.normpath(os.path.join(glob.escape(test_folder_path), '*.gz')).replace('\\', '/'))
            )
        ]

        # loop through all logs inside current test folder
        for test_log_path in test_logs_paths:
            # take only the name of the log file
            test_log_filename = test_log_path.split('/')[-1]
            # split graph's name and log name
            test_log_graph, test_log_name = test_log_filename.split('_', 1)

            if "avg_iter_time_log" in test_log_name or "max_iter_time_log" in test_log_name:
                # avg_iter_time and max_iter_time need to be treated in a particular way since made
                # by tuples and not by single float values
                logs[test_folder_path][test_log_filename] = [
                    tuple([float(s.split(",")[0]), float(s.split(",")[1])]) for s in np.loadtxt(test_log_path, str)
                ]
            elif log_pattern.match(test_log_filename) or test_log_name in ['iter_time.gz', 'iter_time.txt.gz']:
                # load log into dict normally without preprocessing values
                logs[test_folder_path][test_log_filename] = np.loadtxt(test_log_path)

    # get the list of all logs' names of the first test folder without duplicates
    avg_log_names = set(logs[test_folder_paths[0]].keys())
    for i in range(1, len(test_folder_paths)):
        # intersect with all other folders' logs' names
        avg_log_names &= set(logs[test_folder_paths[i]].keys())

    """
    avg_logs = {
        $LOG_X_NAME : [$LOG_X_TEST#1, $LOG_X_TEST#2, ...],
        ...
        $LOG_Y_NAME : [$LOG_Y_TEST#1, $LOG_Y_TEST#2, ...],
        ...
    }
    """
    avg_logs = {}
    min_log_lengths = {} # $LOG_NAME : $MIN_LOG_LENGTH
    new_setup_graphs_names = set([])
    for test_folder_path in logs:
        for log_name in list(avg_log_names):
            new_setup_graphs_names.add(log_name.split('_', maxsplit=1)[0])
            if log_name not in avg_logs:
                avg_logs[log_name] = []
                min_log_lengths[log_name] = math.inf
            avg_logs[log_name].append(logs[test_folder_path][log_name])
            min_log_lengths[log_name] = min(min_log_lengths[log_name], len(logs[test_folder_path][log_name]))

    for log_name in list(avg_log_names):
        avg_logs[log_name] = [l[0:min_log_lengths[log_name]] for l in avg_logs[log_name]]
        avg_logs[log_name] = np.array(np.sum(avg_logs[log_name], axis=0)) / len(avg_logs[log_name])

    new_ordered_setup_graphs_names = []
    for graph in setup['graphs']:
        if graph in list(new_setup_graphs_names):
            new_ordered_setup_graphs_names.append(graph)

    setup['graphs'] = graphs.generate_n_nodes_graphs(setup['n'], new_ordered_setup_graphs_names)
    setup['metrics'] = list(setup_metrics)
    setup['real_metrics'] = list(setup_real_metrics)

    avg_output_dir = os.path.normpath(os.path.join(
        test_folder_paths[0].rsplit('/', maxsplit=1)[0],
        test_folder_paths[0].rsplit('/', maxsplit=1)[1].split('conflict')[0] + 'AVG'
    ))

    if os.path.exists(avg_output_dir):
        if input(
                "Folder {} already exists, continuing will cause the loss of all data already inside it, continue "
                "anyway? (type 'y' or 'yes' to continue or any other key to abort)".format(avg_output_dir)) not in [
            'y', 'yes']:
            raise Exception("Script aborted")
        rmtree(avg_output_dir)
    os.makedirs(avg_output_dir)

    for log_name in avg_logs:
        np.savetxt(os.path.normpath(os.path.join(
            avg_output_dir,
            log_name
        )), avg_logs[log_name], delimiter=',')

    with open(os.path.join(avg_output_dir, '.setup.pkl'), "wb") as f:
        pickle.dump(setup, f, pickle.HIGHEST_PROTOCOL)

    # Fill descriptor with setup dictionary
    descriptor = """>>> Test Descriptor File
    AVERAGE TEST OUTPUT FILE
    Date: {}
    Tests merged: {}\n
    """.format(str(datetime.datetime.fromtimestamp(time.time())),
        pprint.PrettyPrinter(indent=4).pformat(test_folder_paths))

    for k, v in setup.items():
        descriptor += "{} = {}\n".format(k, v)
    descriptor += "\n"

    with open(os.path.join(avg_output_dir, '.descriptor.txt'), "w") as f:
        f.write(descriptor)


if __name__ == '__main__':
    main()
