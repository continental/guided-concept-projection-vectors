# Installation

1. Download repo
```
git clone https://github.com/continental/guided-concept-projection-vectors.git
```

2. Create & activate venv (optionally)

We used Python 3.9.17

```
python -m venv gcpv_venv
source ./gcpv_venv/bin/activate
```

3. Install requirements
```
pip install -r requirements.txt
```


# Demo

## Download MS COCO 2017 annotations + validation subset (240 MB + 780 MB)
Execute: `./data/download_ms_coco_2017val_dataset.sh`

## Try demo Jupyter Notebooks files

1. Optimize GCPVs for a single sample: `./demo/gcpv_optimization.ipynb`
2. Optimize & cluster several GCPVs + weak concept localization: `./demo/gcpv_clustering.ipynb`
3. Find subconcepts with GCPVs + weak sub-concept localization: `./demo/gcpv_clustering_subconcepts.ipynb`



# Reference

ArXiv.org:

```
@article{mikriukov2023gcpv,
  title={GCPV: Guided Concept Projection Vectors for the Explainable Inspection of CNN Feature Spaces},
  author={Mikriukov, Georgii and Schwalbe, Gesina and Hellert, Christian and Bade, Korinna},
  journal={arXiv preprint arXiv:2311.14435},
  year={2023}
}
```



# Documentation

For further help, see the API-documentation or contact the maintainers.



# License

Copyright (C) 2024 co-pace GmbH (a subsidiary of Continental AG). All rights reserved.
