[Documentation - Home](../index.md)

## 1. Installation


Erdre requires a working installation of Python3.8. 
These instructions are specifically for Linux operating systems or Windows Subsystem for Linux (WSL).

Clone this repository:

```
git clone https://github.com/SINTEF-9012/Erdre
```

Enter the cloned repository:

```
cd Erdre/
```

Create a virtual environment (optional):

```
mkdir venv
python3 -m venv venv
source venv/bin/activate
```

Install the requirements:

```
pip3 install -r requirements.txt
```

To get a plot of the neural network architecture, the following software needs
to be installed: [Graphviz](https://graphviz.org/about/).

## Alternative way of installing requirements (not recommended)

As an alternative you can install the required modules by running the command below, but be aware that this may cause problems due to mismatching version requirements.

```
pip3 install dvc pandas pandas-profiling sklearn xgboost tensorflow tensorflow-probability edward2 plotly nonconformist
```


Next: [Quickstart](02_quickstart.md)
