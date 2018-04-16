# TODOs

## Highest priority
- [ ] Introduce time in nodes' task computation.
- [ ] Implement the suggested heavy conditioned function generator.
- [ ] Output the three plots requested by the advisor.
- [ ] Simulate with different dependency graphs to show some result.

## Very high priority
- [x] Utilize python curses module instead of classic print for a better output visualization.
- [x] Change dict setup to function parameters to avoid too much memory consumption.
- [ ] Redesign training set generator functions.
- [ ] Exploit all parameters of mltoolbox objects that are actually unused.
- [ ] Implement different kind of loss functions.
- [ ] Implement several metrics for evaluate the training stage.
- [ ] Implement plotting system.
- [ ] Add stop condition in addition to `max_iter` (e.g. error under a certain threshold).
- [ ] Improve all gradient descent algorithms' performances.
- [ ] Design better generator function for the training_set.

## High priority
- [x] Get rid of all deepcopy functions and consequently improve dataset management.
- [ ] Nice output with ncurses on a new window different from the terminal in which should appear errors and warnings.
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

## Discarded
- [ ] ~~Change everything to SKLearn~~.
- [ ] ~~Augment precision in computation by exploiting Decimal module in std library~~.