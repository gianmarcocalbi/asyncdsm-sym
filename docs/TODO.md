# TODOs

## Highest priority (_EMERGENCIES_)
- [ ] Refactor mtm and Pn everywhere.
- [ ] Check all cluster.run method.

## Very high priority
- [x] Introduce time in nodes' task computation.
- [x] Output the three plots requested by the advisor.
- [x] Simulate with different dependency graphs to show some result.
- [x] Implement the suggested heavy conditioned function generator.
- [x] Implement method to store result on files easily.
- [x] Implement the automatic generation of a desc file within `"test_log/test_XXX"` folders which describes the setup of the test.
- [x] Change how MSE is computed.
- [x] Fix "stop due to `epsilon`" to work with the actual plotting system.
- [x] Getting started with Jupiter Notebook.
- [x] Add parse flags to plotter in order to have a fast plotting method.
- [x] Plotter auto-detects which graphs to plot basing on the logs prefixes in test log folder.
- [x] Same name and color on graphs.
- [x] Fix trailing segment after plotting averaged metrics obtained from a simulation which ended due to `max_time` (the problem occurs only when averaging).  
- [x] Add new cluster parameter to set lambda ratio and distribution of the time taken by each node to perform its task. 

## High priority
- [x] Change dict setup to function parameters to avoid too much memory consumption.
- [x] Redesign training set generator functions.
- [x] Design better generator function for the training_set.
- [x] Implement several metrics for evaluate the training stage.
- [x] Add stop condition in addition to `max_iter` (e.g. error gets under a certain threshold).
- [x] Implement different kind of loss functions.

## Medium priority
- [x] Get rid of all deepcopy functions and consequently improve dataset management.
- [x] Implement `verbose` flag everywhere and in particular in mltoolbox SGD algorithms.
- [x] Implement error handling.
- [x] Try the SGD method with a bigger dataset whose informations are already known (for example accuracy after N steps).


## Medium priority 
- [x] Consider a non-constant learning rate.
- [x] Starting reasoning with Expanders.
- [ ] Comment everything as numpy does.

## Low priority
- [x] Try a different function to minimize: rather than mean square error consider instead something like **cross entropy** .

## Lowest priority

## Maybe
- [ ] Improve all gradient descent algorithms' performances.

## Discarded
- [ ] ~~Change everything to SKLearn~~.
- [ ] ~~Augment precision in computation by exploiting Decimal module in std library~~.
- [ ] ~~Create plotting module~~.
- [ ] ~~Nice output with ncurses on a new window different from the terminal in which should appear errors and warnings~~.
- [ ] ~~Make time taken by a task depends on size of batch (`1` for SGD, `batch_size` for BGD, `self.N` for GD): `lambda = 1 / batch_size`.~~
- [ ] ~~Graphical interface with PyQt~~.
- [ ] ~~Add validation~~.
- [ ] ~~Implement smart recognizing of activation_function based on training_set output values~~.
- [ ] ~~Test SGD correctness with Leon Bottou suggestions~~.
- [ ] ~~Try something different than learning rate (Hessian inverse approximation - SGDQN algorithm or Averaged SGD)~~.