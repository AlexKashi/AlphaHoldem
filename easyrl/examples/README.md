### Hyperparameter Sweep

First, define a yaml file (e.g., `sac_sweeper.yml`) that specifies the search values for each hyperparameter. And run
the following command:

```bash
hpsweep -f sweepers/sac_sweeper.yml
```