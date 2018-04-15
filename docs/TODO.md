# TODOs

## Highest priority
- [ ] Introduce time in nodes' task computation.
- [ ] Implement suggested the heavy conditioned function generator.
- [ ] Output the three plots requested by the advisor.
- [ ] Simulate with different dependency graphs to show some result.

## Very high priority
- [x] Utilize python curses module instead of classic print for a better output visualization.
- [x] Change dict setup to function parameters to avoid too much memory consumption.
- [ ] Implement different kind of loss functions.
- [ ] Implement several metrics for evaluate the training stage.
- [ ] Implement plotting system.
- [ ] Improve all gradient descent algorithms' performances.
- [ ] Design better generator function for the training_set.

## High priority
- [x] Get rid of all deepcopy functions and consequently improve dataset management.
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