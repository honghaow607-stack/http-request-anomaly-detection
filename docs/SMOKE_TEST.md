# Publishing verification

Date: 2026-10-28. These checks verify that the published project can be installed and exercised. They do not replace or independently reproduce the author's reference classifier experiment.

Environment: macOS/Darwin 25.5.0, arm64, Python 3.12.14, OpenMP runtime from Homebrew `libomp` 23.1.0. Main packages: NumPy 2.5.4, pandas 3.1.0, SciPy 1.18.1, scikit-learn 1.9.1, XGBoost 3.4.1 and Streamlit 1.66.1. `requirements-smoke-lock.txt` records this verification environment, not the historical experiment's environment.

| Check | Result |
|---|---|
| Install `requirements.txt` in a fresh virtual environment | Passed |
| `python -m pip check` | Passed |
| Compile Python sources | Passed; original implementation emitted one invalid-escape SyntaxWarning |
| Prepare paper inputs from original local CSIC files | Passed: 26,572 retained requests |
| Download paper inputs from the pinned mirror with content-hash checks | Passed: 26,572 retained requests |
| Compare network-prepared records with original cached dataset | Passed: equal structured records |
| Prepare bundled synthetic-only inputs in a clean copy | Passed: 986 retained requests |
| Run unchanged three-model training, evaluation, plotting and serialization on synthetic data | Passed |
| Run CLI prediction on both bundled example request files | Passed |
| Open Streamlit page and execute its normal-request example | Passed |
| Reject an extra raw file that changes the selected protocol | Passed |
| Reject a modified bundled synthetic file with a hash mismatch | Passed |
| Check imported implementation hashes and README relative links | Passed |

The synthetic model produced during these checks is not included in the repository. The original `results/reference/` artifacts remain unchanged. No full-corpus classifier training or statistical comparison was performed as part of publishing verification.
