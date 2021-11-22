# Table of contents
* [What are the tasks?](#tasks)
* [Who provides the data? How is it collected?](#data-sets)
* [How to run project](#run-project)

## Tasks
The following tasks are given:

<ol start="0">
  <li>Collect and download needed data.</li>
  <li> Explore the data about education and employment rates collected from multiple websites.</li>
  <li>Linear regression.</li>
  <li>RSS vs. polyfit.</li>
  <li>Pairs bootstrap between education and unemployment rates + condifence intervals.</li>
  <li>Plot bootstrap regressions.</li>
  <li>Hypothesis test on Pearson correlation.</li>
  <li>Conclusions.</li>
</ol>

## Data sets
Data sets for unemployment rates and education rates are selected and downloaded from [Eurostat](https://ec.europa.eu/eurostat/web/products-eurostat-news/-/DDN-20190920-1) and [OECD](https://data.oecd.org/unemp/unemployment-rates-by-education-level.htm).

Notice: The websites only provide seperated CSV-files for countries. These data sets need to be merged within the python code.

## Run project
Needed: python3 and pip.

To run the project, do the following:
1. Install all requirements and packages.
<br>(*Notice: Always install packages wihtin a virtual environment to reduce possible harm on the system compiler.*)

```
pip install -r requirements.txt
```

2. Open the `main.ipynb` file or click [here](/examTask/main.ipynb). That is the Jupyter-Notebook project providing all the solutions and documentation for this project.

3. Run the `main.ipynb` file.