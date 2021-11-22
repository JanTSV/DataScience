import pandas as pd
from pathlib import Path
from pandas.core.frame import DataFrame
import matplotlib.pyplot as plt
import numpy as np


def read_raw_data(low_file: Path, middle_file: Path, high_file: Path) -> dict:
    """
    Read the raw data which is always seperated into differen files for low, middle and high education as Dataframe.

    Args:
        low_file: Low education data.
        middle_file: Middle education data.
        high_file: High education data.
    """
    
    ret = {
        "Low": pd.read_csv(low_file),
        "Middle": pd.read_csv(middle_file),
        "High": pd.read_csv(high_file)
    } 
    return ret


def merge_and_clean_data(eurostrat_data: dict, oecd_data: dict) -> DataFrame:
    """
    1. Extract needed data and convert data types.
    2. Merge data into a single normalized data frame.

    Args:
        eurostrat_data: Data from eurostrat.
        oecd_data: Data from OECD.
    """
    ret = pd.DataFrame(columns=["Year", 
                                "USA - Low", "USA - Middle", "USA - High", 
                                "Germany - Low", "Germany - Middle", "Germany - High"])

    # ret["Year"] = ret["Year"].asfreq("A")
    # ret["Year"] = ret["Year"].astype()

    # USA
    ret["USA - Low"] = ret["USA - Low"].astype(float)  # Means: Unemloyed population in % with Low education
    ret["USA - Middle"] = ret["USA - Middle"].astype(float)
    ret["USA - High"] = ret["USA - High"].astype(float)

    # Germany
    ret["Germany - Low"] = ret["Germany - Low"].astype(float)
    ret["Germany - Middle"] = ret["Germany - Middle"].astype(float)
    ret["Germany - High"] = ret["Germany - High"].astype(float)
    
    # Copy time
    ret["Year"] = oecd_data["Low"]["TIME"][oecd_data["Low"]["LOCATION"] == "USA"].astype(int).values

    # Extract USA
    ret["USA - Low"]  = oecd_data["Low"]["Value"][oecd_data["Low"]["LOCATION"] == "USA"].astype(float).values
    ret["USA - Middle"] = oecd_data["Middle"]["Value"][oecd_data["Middle"]["LOCATION"] == "USA"].astype(float).values
    ret["USA - High"] = oecd_data["High"]["Value"][oecd_data["High"]["LOCATION"] == "USA"].astype(float).values

    # Extract and merge Germany
    ret["Germany - Low"] = (eurostrat_data["Low"]["Value"][eurostrat_data["Low"]["GEO"] == "Germany (until 1990 former territory of the FRG)"].astype(float).values + \
                            oecd_data["Low"]["Value"][oecd_data["Low"]["LOCATION"] == "DEU"].astype(float).values) / 2
    ret["Germany - Middle"] = (eurostrat_data["Middle"]["Value"][eurostrat_data["Middle"]["GEO"] == "Germany (until 1990 former territory of the FRG)"].astype(float).values + \
                               oecd_data["Middle"]["Value"][oecd_data["Middle"]["LOCATION"] == "DEU"].astype(float).values) / 2
    ret["Germany - High"] = (eurostrat_data["High"]["Value"][eurostrat_data["High"]["GEO"] == "Germany (until 1990 former territory of the FRG)"].astype(float).values + \
                             oecd_data["High"]["Value"][oecd_data["High"]["LOCATION"] == "DEU"].astype(float).values) / 2

    ret["Year"] = pd.to_datetime(ret["Year"], format="%Y").dt.to_period("Y")
    return ret


def plot_data(df: DataFrame, country: str, show=True):
    plt.xlabel("Year")
    plt.ylabel("Unemployed population in percent")
    df_c = df.copy()
    df_c["Year"] = df_c["Year"].dt.year
    plt.plot("Year", F"{country} - Low", data=df_c, label=F"{country} | Low education")
    plt.plot("Year", F"{country} - Middle", data=df_c, label=F"{country} | Middle education")
    plt.plot("Year", F"{country} - High", data=df_c, label=F"{country} | High education")
    if show: 
        plt.legend()
        plt.show()


def plot_single_data(df: DataFrame, key: str, show=True):
    plt.xlabel("Year")
    plt.ylabel(F"Unemployed population in percent")
    plt.title(key)
    df_c = df.copy()
    df_c["Year"] = df_c["Year"].dt.year
    plt.plot("Year", key, data=df_c)
    if show:
        plt.show()


def pearson_r(x, y):
    """Compute Pearson correlation coefficient between two arrays."""
    # Compute correlation matrix: corr_mat
    corr_mat = np.corrcoef(x, y)

    # Return entry [0,1]
    return corr_mat[0,1]


def plot_linear_regression(x, y, title, show=True):
    plt.xlabel("Year")
    plt.ylabel("Unemployed population in percent") 
    plt.plot(x, y)
    plt.title(title)

    # 1 dimensional ployfit
    a, b = np.polyfit(x, y, 1)
    model_1d = np.poly1d(np.polyfit(x, y, 1))
    plt.plot(x, model_1d(x), linewidth=2, color="red", label="Ployfit | 1D")

    # 3 dimensional ployfit
    model_3d = np.poly1d(np.polyfit(x, y, 3))
    plt.plot(x, model_3d(x), linewidth=2, color="yellow", label="Ployfit | 3D")
    if show:
        print(title, "Slope: ", a, " Intercept: ", b)
        plt.legend()
        plt.show()


def calculate_slope(x, y):
    slope = ((x-x.mean())*(y-y.mean())).sum()/(np.square(x-x.mean())).sum()
    return(slope)


def calculate_interception_point(x, y, m):
    c = y.mean()-(m*x.mean())
    return(c)


def calculate_rss(x, y):
    slope = calculate_slope(x, y)
    intercept = calculate_interception_point(x, y, slope)
    return rss(x, y, slope, intercept)


def rss(x, y, slope, intercept):
    return np.sum(np.square(y - slope * x - intercept))


def slope_vs_rss(x, y):
    a_vals = np.linspace(-1, 1, 100)
    rss_array = np.empty_like(a_vals)
    slope = calculate_slope(x, y)
    intercept = calculate_interception_point(x, y, slope)
    for i, a in enumerate(a_vals):
        rss_array[i] = rss(x, y, a, intercept)
    
    best_rss = rss_array.min()
    best_slope = float(a_vals[np.where(rss_array == best_rss)])
    plt.xlabel("Slope")
    plt.ylabel("RSS")
    plt.plot(a_vals, rss_array)
    plt.plot(best_slope, best_rss, "ro", label=F"Minimum [{best_slope} | {int(best_rss)}]")
    plt.legend()
    plt.show()


def linear_regression_vs_rss(x, y, title, show=True):
    plt.xlabel("Year")
    plt.ylabel("Unemployed population in percent") 
    plt.plot(x, y)
    plt.title(title)

    # 1 dimensional ployfit
    slope, intercept = np.polyfit(x, y, 1)
    model_1d = np.poly1d(np.polyfit(x, y, 1))
    plt.plot(x, model_1d(x), linewidth=2, color="red", label="Ployfit | 1D")

    # RSS
    rss= calculate_rss(x, y)

    # Draw residuals
    points = x.max() - x.min() + 1
    a_vals = np.linspace(x.min(), x.max(), points)
    residuals = np.empty_like(a_vals)

    # Calculate residuals
    for i, a in enumerate(a_vals):
        residuals[i] = y[i] - (slope * a + intercept)

    # Plot residuals
    label = None
    for i, residual in enumerate(residuals):
        if i == 0:
            label = "Residuals"
        else:
            label = None
        plt.plot([a_vals[i], a_vals[i]], [y[i], y[i] - residual], linewidth=0.5, alpha=0.5, color="red", label=label)

    if show:
        plt.legend()
        plt.show() 


def rss_parabola(slope):
    """Helper function for calculating the plotting range of the rss function. It inverts the minimum of the rss function what generates two new minimums that can then be used as bounding points for the plotting of the actual rss function."""
    # First: move the rss function on the y axis to zero and then 10 Percent of this way more into the negative direction.
    # Second: Mirror the negative part (minimum of the function) on the x axis in the positive direction.
    return abs(calculate_rss(x, y, i, intercept) - slopeY * 1.1)



def pairs_bootstrap(x, y, size=1):
    inds = np.arange(len(x))
    slopes = []
    intercepts = []

    for _ in range(size):
        index = np.random.choice(inds, len(inds))
        xx, yy = x[index], y[index]
        slope, intercept = np.polyfit(xx, yy, 1)
        slopes.append(slope)
        intercepts.append(intercept)
    
    return slopes, intercepts


def pearson_coefficient_hypothesis_test(permute_data, fix_data, func=pearson_r, size=1000):
    pearson_observed = func(permute_data, fix_data)
    replicates = np.empty(size)
    print("Observed pearson: ", pearson_observed)

    for i in range(size):
        permutated = np.random.permutation(permute_data)
        replicates[i] = func(permutated, fix_data)
    
    p_value = np.sum(replicates >= pearson_observed) / size
    print("P-Value: ", p_value)
