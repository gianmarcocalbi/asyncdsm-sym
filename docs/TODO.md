# TODOs

## Highest priority (**EMERGENCIES**)

## Very high priority
- [x] Introduce time in nodes' task computation.
- [x] Output the three plots requested by the advisor.
- [x] Simulate with different dependency graphs to show some result.
- [x] Implement the suggested heavy conditioned function generator.
- [x] Implement method to store result on files easily.
- [x] Implement the automatic generation of a desc file within `"test_log/test_XXX"` folders which describes the setup of the test.
- [x] Change how MSE is computed.
- [ ] Make time taken by a task depends on size of batch (`1` for SGD, `batch_size` for BGD, `self.N` for GD): `lambda = 1 / batch_size`.
- [x] Fix "stop due to `epsilon`" to work with the actual plotting system.
- [x] Getting started with Jupiter Notebook.
- [x] Add parse flags to plotter in order to have a fast plotting method.
- [ ] Plotter auto-detects which graphs to plot basing on the logs prefixes in test log folder.
- [ ] Fix trailing segment after plotting averaged metrics obtained from a simulation which ended due to `max_time`.  
- [ ] Test SGD correctness with Leon Bottou suggestions.
- [ ] Add new cluster parameter to set lambda ratio and distribution of the time taken by each node to perform its task. 

## High priority
- [x] Change dict setup to function parameters to avoid too much memory consumption.
- [x] Redesign training set generator functions.
- [x] Design better generator function for the training_set.
- [x] Implement several metrics for evaluate the training stage.
- [x] Add stop condition in addition to `max_iter` (e.g. error gets under a certain threshold).
- [ ] Implement different kind of loss functions.
- [ ] Comment everything as numpy does.

## Medium priority
- [x] Get rid of all deepcopy functions and consequently improve dataset management.
- [ ] Implement `verbose` flag everywhere and in particular in mltoolbox SGD algorithms.

- [ ] Implement error handling.
- [ ] Try the SGD method with a bigger dataset whose informations are already known (for example accuracy after N steps).
- [ ] Add validation.
- [ ] Implement smart recognizing of activation_function based on training_set output values.

## Medium priority 
- [ ] Consider a non-constant learning rate.
- [ ] Starting reasoning with Expanders.

## Low priority
- [ ] Try a different function to minimize: rather than mean square error consider instead something like **cross entropy** .
- [ ] Try something different than learning rate (Hessian inverse approximation - SGDQN algorithm or Averaged SGD).

## Lowest priority
- [ ] Graphical interface with PyQt.

## Maybe
- [ ] Improve all gradient descent algorithms' performances.

## Discarded
- [ ] ~~Change everything to SKLearn~~.
- [ ] ~~Augment precision in computation by exploiting Decimal module in std library~~.
- [ ] ~~Create plotting module~~.
- [ ] ~~Nice output with ncurses on a new window different from the terminal in which should appear errors and warnings~~.