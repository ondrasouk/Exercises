# MPC-MLF-project
MPC-MLF final project

## How to start

1) Clone the repository

2) Install the dependencies
```bash
python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## How to run

Run the program and print help
```bash
python main.py -h
```
Read the help!

## Features

 - Load dataset from CSV files and save it in more efficient format of compressed numpy arrays.
 - Save/load model data and training history
 - Export prediction of test data to CSV file
 - Plot training history to PGFplots files and show it.
