

How to put the files?

Files structure

```shell
M44TMD_Repository
├── code
│   ├── dataloader.py
│   ├── train.py
│   └── test.py
├── data
│   ├── annotation
│   │   ├── train.txt
│   │   └── test.txt
│   └── image
├── requirements.txt
└── README.md
```

How to run the code?

```shell
cd M44TMD/code
```

To train the model: 

```python
python train.py
```

To test the model: 

```python
python inference.py
```
