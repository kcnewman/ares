

```
ares
├─ .python-version
├─ Dockerfile
├─ LICENSE
├─ README.md
├─ app.py
├─ artifacts
│  ├─ cache
│  │  ├─ geocode_cache.json
│  │  └─ schema.json
│  ├─ data
│  │  └─ raw.csv
│  ├─ data_processing
│  │  ├─ preprocessed_eval.csv
│  │  └─ preprocessed_train.csv
│  ├─ data_split
│  │  ├─ eval.csv
│  │  └─ train.csv
│  ├─ data_validation
│  │  └─ status.txt
│  ├─ feature_engineering
│  │  ├─ class_pi.json
│  │  ├─ features_test.csv
│  │  ├─ features_train.csv
│  │  ├─ global_lux_median.json
│  │  ├─ global_ref.json
│  │  ├─ loc_iqr.json
│  │  ├─ loc_luxury_median.json
│  │  ├─ loc_pi.json
│  │  └─ locality_stats.json
│  ├─ model_evaluation
│  │  └─ metrics.json
│  └─ model_trainer
│     └─ model.joblib
├─ config
│  └─ config.yaml
├─ experiments
│  └─ mlflow.db
├─ logs
├─ main.py
├─ params.yaml
├─ pyproject.toml
├─ reports
│  ├─ amenity_on_price.png
│  ├─ avg_price_loc.png
│  ├─ cond_analysis.png
│  ├─ corr.png
│  ├─ furn_analysis.png
│  ├─ hse_type_analysis.png
│  ├─ loc_dist.png
│  ├─ outliers.png
│  ├─ price_by_loc.png
│  ├─ price_dis_hse_condition.png
│  ├─ price_dis_hse_type.png
│  ├─ price_dist.png
│  └─ price_vs_feats.png
├─ schema.yaml
├─ src
│  └─ ares
│     ├─ __init__.py
│     ├─ api
│     │  └─ main.py
│     ├─ components
│     │  ├─ __init__.py
│     │  ├─ data_processing.py
│     │  ├─ data_split.py
│     │  ├─ data_validation.py
│     │  ├─ feature_engineering.py
│     │  ├─ model_evaluation.py
│     │  └─ model_trainer.py
│     ├─ config
│     │  ├─ __init__.py
│     │  └─ configuration.py
│     ├─ constants
│     │  └─ __init__.py
│     ├─ entity
│     │  ├─ __init__.py
│     │  └─ config_entity.py
│     ├─ pipeline
│     │  ├─ __init__.py
│     │  ├─ data_processing.py
│     │  ├─ data_split.py
│     │  ├─ data_validation.py
│     │  ├─ feature_engineering.py
│     │  ├─ inference.py
│     │  ├─ model_evaluation.py
│     │  └─ model_trainer.py
│     └─ utils
│        ├─ __init__.py
│        └─ common.py
├─ template.py
├─ tests
│  ├─ conftest.py
│  ├─ logs
│  ├─ test_data_processing.py
│  ├─ test_data_splitting.py
│  ├─ test_data_validation.py
│  └─ test_feature_engineering.py
└─ uv.lock

```