"""
Jenny Hoang, Helen Jiang
CSE 163
Final Project - Data Preparation

This program prepares raw datasets for the college outcomes EDA.
It cleans the College Scorecard, cleans the Census ACS median income
data, and filters the SHEEO data to FY2024. Run this before eda.py.
Output files are saved to the data/ directory.
"""


import pandas as pd

SCORECARD_COLS = [
    "INSTNM",           # College name
    "STABBR",           # State
    "CONTROL",          # Public (1) / Private nonprofit (2)
    "C150_4",           # 6-year graduation rate
    "MD_EARN_WNE_P10",  # Median earnings 10 years after entry
]


SHEEO_COLS = ['State', 'Education Appropriations', 'Net FTE Enrollment']


def clean_college_scorecard_data() -> None:
    """
    Load the College Scorecard, keep only needed columns, filter to
    public and private nonprofit colleges, drop rows with missing
    graduation rate or earnings, and save the cleaned CSV.
    """

    # Read in the data
    df = pd.read_csv("data/Most-Recent-Cohorts-Institution.csv",
                     usecols=SCORECARD_COLS,
                     na_values=['NULL', 'PrivacySuppressed'],
                     low_memory=False)

    # Filter to only public and private nonprofit colleges
    df = df[df["CONTROL"].isin([1, 2])]

    # Drop rows with missing values in the columns we care about
    df = df.dropna(subset=["C150_4", "MD_EARN_WNE_P10"])

    # Save the cleaned dataset
    df.to_csv("data/college_scorecard_clean.csv", index=False)
    print(f"Cleaned College Scorecard data saved. Shape: {df.shape}")


def clean_census_data() -> None:
    """
    Load the data and create a simple CSV with each
    state's median income
    """
    # Read the raw Census data
    raw = pd.read_csv('data/Median-Income-Past-12-Months.csv')

    # Select the row labeled 'Households'
    households = raw[raw.iloc[:, 0].str.strip() == 'Households']

    # Find all columns that contain median income estimates
    median_cols = [
        c for c in raw.columns
        if 'Median income (dollars)!!Estimate' in c
    ]

    # Get the data for that row
    row = households[median_cols].iloc[0]

    states = row.index.str.split('!!').str[0]
    incomes = row.str.replace(',', '', regex=False).apply(pd.to_numeric,
                                                          errors='coerce')

    df = pd.DataFrame({'State': states.values, 'MedianIncome': incomes.values})
    df = df.dropna(subset=['MedianIncome'])
    df['MedianIncome'] = df['MedianIncome'].astype(int)

    # Save to a new clean CSV
    df.to_csv('data/census_median_income.csv', index=False)
    print(f'Census cleaned and saved. Shape: {df.shape}')


def convert_sheeo_xlsx_to_csv() -> None:
    """
    Read the SHEEO Excel file, extract the 'Report Data' sheet,
    and save as CSV.
    """
    # Read the 'Report Data' sheet from the Excel file
    df = pd.read_excel('data/SHEEO_SHEF_FY24_Report_Data.xlsx',
                       sheet_name='Report Data')

    # Save as CSV
    df.to_csv('data/SHEEO_SHEF_FY24_Report_Data.csv', index=False)
    print(f"Converted successfully! Shape: {df.shape}")


def filter_sheeo_data() -> None:
    """
    Filter the SHEEO data to only include FY2024 rows and keep only the
    columns needed for merging with the College Scorecard data.
    """
    # Read the CSV file
    df = pd.read_csv('data/SHEEO_SHEF_FY24_Report_Data.csv',
                     na_values=['NULL', 'PrivacySuppressed'],
                     low_memory=False)

    # Filter to only FY2024 data
    df_2024 = df[df['FY'] == 2024]

    # Keep only columns we need for merging
    df_2024 = df_2024[SHEEO_COLS]

    # Save the filtered dataset
    df_2024.to_csv('data/SHEEO_FY2024.csv', index=False)
    print(f"Filtered to FY2024 data. Shape: {df_2024.shape}")


def main():
    clean_college_scorecard_data()
    clean_census_data()
    convert_sheeo_xlsx_to_csv()
    filter_sheeo_data()


if __name__ == "__main__":
    main()
