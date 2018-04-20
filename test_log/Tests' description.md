## TEST A
Made before the first conf call between Nizza and Milan Bicocca.

It exploits a function that is no more available, is within the old program structure (right after _reengineering merge_).

### Test A setup
_See report for first conf call._

 
## TEST A2
Attempt (quite successful) to reproduce TEST A outputs within the new program structure by providing also the _Real MSE_ metric.

To generate the training set it exploits `generate_samples_from_function_old` function of mltoolbox.

### Test A2 setup

```python
n =20
seed = 2
X, y = mltoolbox.sample_from_function_old(
    10000, 100, mltoolbox.LinearYHatFunction.f,
    domain_radius=10,
    domain_center=0,
    subdomains_radius=2,
    error_mean=0,
    error_std_dev=1,
    error_coeff=1
)
cluster.setup(
    X, y, mltoolbox.LinearYHatFunction,
    max_iter=4000,
    method="stochastic",
    batch_size=20,
    activation_func=None,
    loss=mltoolbox.SquaredLossFunction,
    penalty='l2',
    epsilon=0.01,
    alpha=0.0005,
    learning_rate="constant",
    metrics="all",
    shuffle=True,
    verbose=False
)
```

##TEST B
???
