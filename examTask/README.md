# Table of contents
* [What are the tasks?](#tasks)
* [Who provides the data? How is it collected?](#data-sets)
* [How to run project](#run-project)

## Tasks
The following tasks are given:

1. Explore the data about education and employment rates collected from multiple websites.
    1. Extract the needed (real) data out of the raw data sets.
    2. Visualize data with e.g. histogram, mean, box-plots, ...
    3. Pearson correlation coefficient.
2. Linear regression.
3. RSS vs. polyfit.
4. Pairs bootstrap between education and unemployment rates + condifence intervals.
5. Plot bootstrap regressions.
6. Hypothesis test on Pearson correlation.
7. Conclusions.
8. Check conclusion against known literature.

## Data sets
Data sets for unemployment rates and education rates are selected and downloaded from [here](https://ec.europa.eu/eurostat/web/products-eurostat-news/-/DDN-20190920-1) and [here](https://data.oecd.org/unemp/unemployment-rates-by-education-level.htm).

Notice: The websites only provide seperated CSV-files for countries. These data sets need to be merged with python-code.

## Run project
Needed: python3 and pip.

To run the project, do the following:
1. Install all requirements and packages:
```
pip install -r requirements.txt
```
This will install all the needed python packages.
Notice: Always install packages wihtin a virtual environment to reduce possible harm on the system compiler.

2. Open the `main.ipynb` file. That is the Jupyter-Notebook project providing all the solutions and documentation for this project.

3. Run the `main.ipynb` file.