# Data

| dataset | source | file |
|---|---|---|
| WDBC | `sklearn.datasets.load_breast_cancer` | (built in) |
| Ionosphere | UCI | `ionosphere.csv` |
| Sonar | UCI (run, but excluded from the final design) | `sonar.csv` |
| Blood Transfusion | UCI | `transfusion.csv` |
| Synthetic | `make_classification(600, 30, n_informative=4, n_redundant=0, n_clusters_per_class=2, flip_y=0.05, random_state=0)` | (generated) |

`python data/download_data.py` downloads the UCI originals into `uci/` and checks that they are identical to the copies in `mirror/`, which are kept for offline use.
When `uci/` is complete the code uses it automatically; otherwise it uses `mirror/`. Which one was used is written to `environment.json` (in `results/run_info/` for the reported run).
H2 and H6 use additional generated data (see `experiments/run_h2.py`).
