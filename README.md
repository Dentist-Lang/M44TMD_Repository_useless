

How to put the files?

Files structure

```shell
M44TMD_Repository
├── Code
│   ├── Models
│   │   ├── M44TMD_ResNet50_fea-fus.py
│   │   └── M44TMD_ResNet50_dec-fus.py
│   ├── loader
│   │   └── dataloader.py
│   └── README.md
├── Train_test_splits
│   ├── Internal_sets
│   │   ├── Internal_training_set.txt
│   │   └── Internal_test_set.txt
│   ├── Temporal_sets
│   │   ├── Temporal_training_set.txt
│   │   └── Temporal_test_set.txt
│   └── External_set
│       └── External_test_set.txt
└── requirements.txt
```

How to run the code?
```shell
cd M44TMD/Code/Models
```

```python
python M44TMD_ResNet50_dec-fus.py
python M44TMD_ResNet50_fea-fus.py
```
