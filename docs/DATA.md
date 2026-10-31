# Data acquisition and exact evaluation scope

## CSIC 2010

Run `python tools/prepare_data.py`. This downloads only `normalTrafficTest.txt` and `anomalousTrafficTest.txt` from the public TF-IDS-Detection mirror and copies the five fixed files in `data/synthetic/`. Dataset authors: C. Torrano Giménez, A. Pérez Villegas and G. Álvarez Marañón. The original dataset documentation is at https://petescully.co.uk/wp-content/uploads/2018/04/http_dataset_csic_2010.pdf .

The study does not include `normalTrafficTrain.txt`. Adding it or regenerating synthetic files changes the corpus. The seven files contain 62,165 parsed requests; canonical-header deduplication retains 26,646 (10,085 normal and 16,561 anomalous). `docs/result_audit.json` records counts and cache checks. Raw requests, the processed cache and the 111 MB trained pickle are excluded from regular Git; retrain the saved bundle with `python run.py`.

## ATRDF 2023

Author-maintained source and download links: https://github.com/ArielCyber/Cisco_Ariel_Uni_API_security_challenge . Dataset citation: Shmuel Lavian and Ariel University, Ariel Cyber Innovation Center (ACIC), *The API Traffic Research Dataset Framework (ATRDF)*, 2023.

Download `dataset_1_train.7z` through `dataset_4_train.7z` from its `Datasets/` directory. Extract each JSON into:

```
data/raw/atrdf/extracted/dataset_1_train/dataset_1_train.json
data/raw/atrdf/extracted/dataset_2_train/dataset_2_train.json
data/raw/atrdf/extracted/dataset_3_train/dataset_3_train.json
data/raw/atrdf/extracted/dataset_4_train/dataset_4_train.json
```

For example, after `pip install py7zr`, extract the archive into the matching directory using a 7z extractor and check the resulting JSON path. The original competition validation files are unlabeled and are not used as labeled tests; absence of Attack_Tag in those files must not be interpreted as evidence that all records are benign. The loader uses Attack_Tag only in the labeled training files.

Dataset 3 is repartitioned 64/16/20 after deduplication. Cross-subset evaluation trains on its 64% partition and evaluates datasets 1, 2 and 4 separately. Different files are not proof of zero leakage: the supplied template-key overlaps are nonzero (see `results/atrdf/atrdf_template_overlap.csv`). Respect upstream dataset terms and attribution; no third-party dataset license is replaced by this repository.
