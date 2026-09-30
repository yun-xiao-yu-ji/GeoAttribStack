# GeoAttribStack

GeoAttribStack is an attribution-regime constrained stacking framework for
geoscientific spatial prediction and uncertainty quantification. It converts
sample-level SHAP profiles into normalized attribution states, estimates soft
regime memberships, and maps those memberships to non-negative, sum-to-one
expert weights. The resulting prediction is an auditable convex combination of
the current base-expert predictions.

## Repository contents

- `geoattribstack/`: lightweight reusable implementation of the fusion core.
- `examples/quick_test.py`: self-contained synthetic-data quick test.
- `tests/test_quick.py`: automated version of the quick test.
- `src/`: complete scripts used for spatial validation, attribution-regime
  fitting, the SHAP correspondence experiment, spatial cross-conformal
  prediction, sensitivity analysis, metric summaries, and figures.
- `config/experiment_config.json`: fixed final-experiment settings.
- `docs/data_dictionary.csv`: definitions for the processed modelling columns.
- `requirements-quick.txt`: dependencies needed only for the quick test.
- `requirements.txt`: pinned environment for the complete experiment.

All source files are provided individually and can be downloaded anonymously
from the repository. No proprietary software is required.

## Quick test

Python 3.10 or 3.11 is recommended. The quick test creates synthetic base-model
predictions and SHAP profiles, fits the attribution regimes and simplex weights,
and verifies that:

1. soft regime memberships sum to one;
2. regime and sample-specific expert weights are non-negative and sum to one;
3. fused predictions remain inside the convex hull of the base predictions; and
4. attribution-conditioned fusion improves over equal weighting in the
   controlled synthetic example.

Create an environment and run the test from the repository root:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements-quick.txt
python examples/quick_test.py
```

Linux or macOS:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-quick.txt
python examples/quick_test.py
```

Expected final line:

```text
QUICK TEST PASSED
```

The automated test can also be run after installing `pytest`:

```bash
python -m pip install pytest
python -m pytest tests/test_quick.py -q
```

## Full experiment

Install the complete pinned environment:

```bash
python -m pip install -r requirements.txt
```

The processed modelling table and locked spatial-fold inputs are distributed in
the public reproducibility archive associated with the manuscript. Place the
processed table at `data/processed/half_ccv_172507_1000_2.csv` and the locked
fold inputs at `locked_arc_input/`.

The primary commands are:

```bash
python src/run_routes.py --input data/processed/half_ccv_172507_1000_2.csv --output work/base_routes --outer-folds 5 --inner-folds 3 --optimization-trials 10 --seed 42 --gate-seeds 11 42 73

python src/run_geoattribstack_arc.py --input data/processed/half_ccv_172507_1000_2.csv --source-cache work/base_routes/route2_validation_audit/results/fold_cache --output work/geoattribstack_arc --folds 5 --seed 42

python src/run_arc_shap_permutation.py --arc-root work/geoattribstack_arc --repeats 10 --seed 20260924

python src/run_final_uncertainty_runtime.py --arc-root work/geoattribstack_arc --output work/final_experiment --seed 42

python src/run_robust_spatial_conformal.py --arc-root work/geoattribstack_arc --output work/final_experiment --seed 42
```

The full workflow is computationally intensive. The quick test is the recommended
first check for installation and algorithmic constraints.

## Data and outputs

The Git repository contains source code and lightweight documentation. Processed
research data, locked fold inputs, machine-readable outputs, and publication
figures are kept in the separate public reproducibility archive to avoid storing
large derived files in Git history. Third-party data products remain subject to
their original provider terms.

## License

The source code and original software documentation are released under the MIT
License. See `LICENSE`.

## Citation

Citation metadata are provided in `CITATION.cff`. Please cite the software
repository and the associated manuscript when using GeoAttribStack.
