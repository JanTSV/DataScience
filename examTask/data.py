import pandas as pd
from pathlib import Path

from pandas.core.frame import DataFrame


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
    ret = pd.DataFrame(columns=["Country", "Year", "Unemployment", "Education"])
    ret["Country"] = ret["Country"].astype(str)
    ret["Year"] = ret["Country"].astype(int)
    ret["Unemployment"] = ret["Country"].astype(float)
    ret["Education"] = ret["Country"].astype(int)
    
    education = {"Low": 0, "Middle": 1, "High": 2}

    # USA
    for key in oecd_data:
        df = oecd_data[key]
        df = df.loc[df["LOCATION"] == "USA"]
        df = df.rename(columns={"LOCATION": "Country", "TIME": "Year", "Value": "Unemployment"})
        df["Education"] = [0] * len(df)
        ret.append(df)
        print(ret)
    return ret