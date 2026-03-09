# Do Public and Private Colleges Produce Different Student Outcomes?
**Authors:** Jenny Hoang, Helen Jiang

**Course:** CSE 163

## Required Libraries
Install the following Python libraries before running:
pip install pandas seaborn matplotlib scipy openpyxl

## Data Downloads
Download the following datasets and place them in a folder called data/:

1. College Scorecard: https://collegescorecard.ed.gov/data/
   Download "Most Recent Institution-Level Data" and rename to:
   Most-Recent-Cohorts-Institution.csv

2. Census ACS Median Income: https://data.census.gov/
   Search "S1903", select all states, download CSV and rename to:
   Median-Income-Past-12-Months.csv

3. SHEEO State Higher Education Finance: https://shef.sheeo.org/data-downloads/
   Download "2024 Report Data" and rename to:
   SHEEO_SHEF_FY24_Report_Data.xlsx

## File Descriptions
- main.py: Cleans and prepares all three raw datasets. Run this first.
- eda.py: Performs the full analysis, generates all visualizations,
  and runs statistical tests.
- test_eda.py: Tests the key functions in eda.py using small
  hand-crafted DataFrames.

## How to Run
Step 1: Download all datasets and place them in the data/ folder.

Step 2: Create an images/ folder in the project directory.

Step 3: Run the cleaning script first: python main.py
   
Step 4: Run the analysis script: python eda.py
   
Step 5: To run tests: python test_eda.py

All output images will be saved to the images/ folder.
