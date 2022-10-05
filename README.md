# pybann
Python Basic Artificial Neural Network

## Install

Create a virtual environment in the cloned repo 

```shell
python3 -m venv pybann-env
```

Activate the virtual environment
```shell
source pybann-env/bin/activate
```

Install requirements
```shell
pip install -r requirements.txt
```

(Optional) Run tests
```shell
python -m unittest discover tests
```

Build and install
```shell
python -m build
pip install dist/pybann-0.1.0.tar.gz
```

(Optional) Build docs
```shell
pdoc pybann -d numpy -o docs/
```

## Dataset

- IRIS (from UCI Machine Learning Repository https://archive.ics.uci.edu/ml/datasets/Iris)

## Resources

Dua, D. and Graff, C. (2019). [UCI Machine Learning Repository](http://archive.ics.uci.edu/ml). Irvine, CA: University of California, School of Information and Computer Science.