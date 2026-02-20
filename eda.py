"""
Jenny Hoang, Helen Jiang
CSE 163
Final Project - Exploratory Data Analysis

This program performs exploratory data analysis on college outcomes data
to examine differences in graduation rates and earnings between public
and private nonprofit colleges, the effect of state income level, and
the relationship between state funding and public college graduation rates.
Run this after main.py.
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

ABBR_TO_STATE = {
    'AL': 'Alabama', 'AK': 'Alaska', 'AZ': 'Arizona',
    'AR': 'Arkansas', 'CA': 'California', 'CO': 'Colorado',
    'CT': 'Connecticut', 'DE': 'Delaware',
    'DC': 'District of Columbia', 'FL': 'Florida',
    'GA': 'Georgia', 'HI': 'Hawaii', 'ID': 'Idaho',
    'IL': 'Illinois', 'IN': 'Indiana', 'IA': 'Iowa',
    'KS': 'Kansas', 'KY': 'Kentucky', 'LA': 'Louisiana',
    'ME': 'Maine', 'MD': 'Maryland', 'MA': 'Massachusetts',
    'MI': 'Michigan', 'MN': 'Minnesota', 'MS': 'Mississippi',
    'MO': 'Missouri', 'MT': 'Montana', 'NE': 'Nebraska',
    'NV': 'Nevada', 'NH': 'New Hampshire', 'NJ': 'New Jersey',
    'NM': 'New Mexico', 'NY': 'New York',
    'NC': 'North Carolina', 'ND': 'North Dakota',
    'OH': 'Ohio', 'OK': 'Oklahoma', 'OR': 'Oregon',
    'PA': 'Pennsylvania', 'RI': 'Rhode Island',
    'SC': 'South Carolina', 'SD': 'South Dakota',
    'TN': 'Tennessee', 'TX': 'Texas', 'UT': 'Utah',
    'VT': 'Vermont', 'VA': 'Virginia', 'WA': 'Washington',
    'WV': 'West Virginia', 'WI': 'Wisconsin', 'WY': 'Wyoming',
}


def load_scorecard(path: str) -> pd.DataFrame:
    """
    Load the pre-cleaned Scorecard CSV. Maps CONTROL to labels and
    adds a STATE_NAME column for merging. Returns a DataFrame.
    """

    df = pd.read_csv(path)
    df['CONTROL'] = df['CONTROL'].map({1: 'Public', 2: 'Private Nonprofit'})
    df['STATE_NAME'] = df['STABBR'].map(ABBR_TO_STATE)
    return df


def load_census(path: str) -> pd.DataFrame:
    """
    Load the pre-cleaned Census CSV. Returns a DataFrame with 'State' and
    'MedianIncome' columns.
    """
    df = pd.read_csv(path)
    return df


def load_sheeo(path: str) -> pd.DataFrame:
    """
    Load the pre-cleaned SHEEO CSV and computes
    FundingPerStudent as Education Appropriations divided by Net FTE
    Enrollment.
    Returns the resulting DataFrame.
    """
    df = pd.read_csv(path)
    df['FundingPerStudent'] = (
        df['Education Appropriations'] / df['Net FTE Enrollment']
    )
    return df


def report_missing(df: pd.DataFrame, name: str) -> None:
    """
    Prints a missingness summary for each column in the given DataFrame.
    """
    print(f'\nMissing data: {name}')
    missing = df.isna().sum()
    missing = missing[missing > 0]
    if missing.empty:
        print('  No missing values detected.')
    else:
        total = len(df)
        for col, n in missing.items():
            print(f'  {col}: {n} missing ({100 * n / total:.1f}%)')


def seven_number_summary(series: pd.Series, label: str) -> None:
    """
    Prints a seven-number summary (mean, std, min, Q1, median, Q3, max)
    for the given numeric Series, labeled with the given label string.
    """
    print(f'\n  {label}:')
    print(f'    Mean={series.mean():.2f}, Std={series.std():.2f}')
    print(
        f'    Min={series.min():.2f}, Q1={series.quantile(0.25):.2f},'
        f' Median={series.median():.2f},'
        f' Q3={series.quantile(0.75):.2f}, Max={series.max():.2f}'
    )


def categorical_summary(series: pd.Series, label: str) -> None:
    """
    Prints the value counts for each unique value in the given
    categorical Series, labeled with the given label string.
    """
    print(f'\n  {label}:')
    counts = series.value_counts()
    print(counts.to_string())


def plot_bar_by_control(scorecard: pd.DataFrame) -> None:
    """
    Figure 1: a bar plot comparing mean graduation rate by college
    type (Public vs Private Nonprofit).
    Takes the cleaned Scorecard DataFrame as input.
    """
    sns.set_theme()
    sns.catplot(
        data=scorecard,
        x='CONTROL',
        y='C150_4',
        kind='bar',
        hue='CONTROL',
        errorbar=None
    )

    plt.xlabel('College Type')
    plt.ylabel('Mean 6-Year Graduation Rate')
    plt.title('Mean Graduation Rate by College Type')

    plt.savefig('images/grad_rate_by_control.png', bbox_inches='tight')
    plt.close()


def plot_earnings_by_control(scorecard: pd.DataFrame) -> None:
    """
    Figure 2: a box plot comparing mean post-graduation earnings
    by college type (Public vs Private Nonprofit).
    Takes the cleaned Scorecard DataFrame as input.
    """
    sns.set_theme()

    plt.figure(figsize=(8, 6))
    sns.boxplot(
        data=scorecard,
        x='CONTROL',
        y='MD_EARN_WNE_P10'
    )

    plt.xlabel('College Type')
    plt.ylabel('Median Earnings 10 Years After Entry (USD)')
    plt.title('Post-Graduation Earnings by College Type')

    plt.savefig('images/earnings_by_control.png', bbox_inches='tight')
    plt.close()


def plot_grad_rate_distribution(scorecard: pd.DataFrame) -> None:
    """
    Figure 3: Distribution of six-year graduation rates (C150_4)
    across all colleges.
    Takes the cleaned Scorecard DataFrame as input.
    """
    sns.set_theme()

    plt.figure(figsize=(8, 6))
    sns.histplot(
        scorecard['C150_4'],
        bins=30,
        kde=True
    )

    plt.xlabel('6-Year Graduation Rate')
    plt.ylabel('Number of Colleges')
    plt.title('Distribution of College Graduation Rates')

    plt.savefig('images/grad_rate_distribution.png', bbox_inches='tight')
    plt.close()


def merge_scorecard_census(scorecard: pd.DataFrame,
                           census: pd.DataFrame) -> pd.DataFrame:
    """
    Merges the Scorecard and Census DataFrames on state name.
    Assigns each college an IncomeLevel category based on the
    median income of its state.
    Returns the merged DataFrame with an added 'IncomeLevel' column.
    """
    merged = scorecard.merge(
        census,
        left_on='STATE_NAME',
        right_on='State',
        how='inner')

    income_levels = merged['MedianIncome'].quantile([1/3, 2/3])
    lower, upper = income_levels.iloc[0], income_levels.iloc[1]

    merged['IncomeLevel'] = pd.cut(
        merged['MedianIncome'],
        bins=[-float('inf'), lower, upper, float('inf')],
        labels=['Low Income', 'Medium Income', 'High Income']
    )

    return merged


def merge_scorecard_sheeo(scorecard: pd.DataFrame,
                          sheeo: pd.DataFrame) -> pd.DataFrame:
    """
    Filters the Scorecard to public colleges,
    then merges with the SHEEO DataFrame on state name,
    returning a new DataFrame.
    """
    public_colleges = scorecard[scorecard['CONTROL'] == 'Public']

    # Calculate state-level averages for graduation rate and earnings
    state_avg = (public_colleges
                 .groupby('STATE_NAME')
                 .agg(
                     AvgGradRate=('C150_4', 'mean'),
                     AvgEarnings=('MD_EARN_WNE_P10', 'mean')
                    )
                 .reset_index()
                 )

    # Merge with SHEEO funding data
    merged = state_avg.merge(
        sheeo,
        left_on='STATE_NAME',
        right_on='State',
        how='inner'
    )
    return merged


def verify_test_assumptions(scorecard, merged_census, merged_sheeo):
    """
    Verifies assumptions for the t-test (RQ1), ANOVA (RQ2),
    and Pearson correlation (RQ3) by checking sample sizes,
    then runs the actual tests.
    """
    public = scorecard[scorecard['CONTROL'] == 'Public']['C150_4']
    private = scorecard[scorecard['CONTROL'] == 'Private Nonprofit']['C150_4']

    # Check assumptions
    print('\nRQ1 T-Test Assumption Checks:')
    print(f'  Public sample size: {len(public)}')
    print(f'  Private sample size: {len(private)}')
    print('  Both > 30, normally distributed')

    # Run the actual t-test
    t, p = stats.ttest_ind(public, private)
    print(f'  T-statistic: {t:.3f}, P-value: {p:.4f}')
    if p < 0.05:
        print('  Result: Significant difference between public and private')
    else:
        print('  Result: No significant difference found')

    low = merged_census[merged_census['IncomeLevel'] == 'Low Income']['C150_4']
    mid = merged_census[merged_census['IncomeLevel']
                        == 'Medium Income']['C150_4']
    high = merged_census[merged_census['IncomeLevel']
                         == 'High Income']['C150_4']

    # Check assumptions
    print('\nRQ2 ANOVA Assumption Checks:')
    print(f'  Low income size: {len(low)}')
    print(f'  Medium income size: {len(mid)}')
    print(f'  High income size: {len(high)}')
    print('  All > 30, normally distributed')

    # Run the actual ANOVA
    f, p = stats.f_oneway(low, mid, high)
    print(f'  F-statistic: {f:.3f}, P-value: {p:.4f}')
    if p < 0.05:
        print('  Result: Significant difference across income levels')
    else:
        print('  Result: No significant difference found')

    # Check assumptions
    print('\nRQ3 Pearson Correlation Assumption Checks:')
    print(f'  Number of states: {len(merged_sheeo)}')
    print('  Both variables are continuous, appropriate for Pearson')

    # Run the actual correlation
    r, p = stats.pearsonr(
        merged_sheeo['FundingPerStudent'], merged_sheeo['AvgGradRate']
    )
    print(f'  r={r:.3f}, P-value: {p:.4f}')
    if p < 0.05:
        print('  Result: Significant correlation between funding '
              'and grad rate')
    else:
        print('  Result: No significant correlation found')


def plot_grad_rate_by_income_level(merged_census: pd.DataFrame) -> None:
    """
    Figure 4: a bar plot comparing mean graduation rate across
    state income levels (Low, Medium, High).
    Takes the Scorecard+Census merged DataFrame as input.
    """
    sns.set_theme()
    level_order = ['Low Income', 'Medium Income', 'High Income']

    plt.figure(figsize=(8, 6))
    sns.barplot(
        data=merged_census,
        x='IncomeLevel',
        y='C150_4',
        order=level_order,
        errorbar=None
    )

    plt.xlabel('State Income Level')
    plt.ylabel('Mean 6-Year Graduation Rate')
    plt.title('Mean Graduation Rate by State Income Level')

    plt.savefig('images/grad_rate_by_income_level.png', bbox_inches='tight')
    plt.close()


def plot_income_vs_grad_rate(merged_census: pd.DataFrame) -> None:
    """
    Figure 5: Scatter plot with regression line showing the relationship
    between state median household income and college graduation rate.
    Takes the Scorecard+Census merged DataFrame as input.
    """
    sns.set_theme()

    plt.figure(figsize=(8, 6))
    sns.regplot(
        data=merged_census,
        x='MedianIncome',
        y='C150_4',
        scatter_kws={'alpha': 0.6}
    )

    plt.xlabel('State Median Household Income (USD)')
    plt.ylabel('6-Year Graduation Rate')
    plt.title('State Median Income vs College Graduation Rate')

    plt.savefig('images/income_vs_grad_rate.png', bbox_inches='tight')
    plt.close()


def plot_funding_vs_grad_rate(merged_sheeo: pd.DataFrame) -> None:
    """
    Figure 6: a scatter plot with regression line showing the
    relationship between per-student state education funding and average
    public college graduation rate by state.
    Takes the Scorecard+SHEEO merged DataFrame as input.
    """
    sns.set_theme()

    plt.figure(figsize=(8, 6))
    sns.regplot(
        data=merged_sheeo,
        x='FundingPerStudent',
        y='AvgGradRate',
        scatter_kws={'alpha': 0.7}
    )

    plt.xlabel('State Funding per Student (USD)')
    plt.ylabel('Average Public College Graduation Rate')
    plt.title('State Funding per Student vs Public College Graduation Rate')

    plt.savefig('images/funding_vs_grad_rate.png', bbox_inches='tight')
    plt.close()


def plot_grad_rate_by_funding_level(merged_sheeo: pd.DataFrame) -> None:
    """
    Figure 7: Bar plot comparing average public college graduation rates
    across low, medium, and high state funding levels assigned by
    tertiles of FundingPerStudent.
    Takes the Scorecard+SHEEO merged DataFrame as input.
    """
    sns.set_theme()

    df = merged_sheeo.copy()
    df['FundingLevel'] = pd.qcut(
        df['FundingPerStudent'],
        q=3,
        labels=['Low Funding', 'Medium Funding', 'High Funding']
    )

    plt.figure(figsize=(8, 6))
    sns.barplot(
        data=df,
        x='FundingLevel',
        y='AvgGradRate',
        order=['Low Funding', 'Medium Funding', 'High Funding'],
        errorbar=None
    )

    plt.xlabel('State Funding Level')
    plt.ylabel('Average Public College Graduation Rate')
    plt.title('Graduation Rate by State Funding Level')

    plt.savefig('images/grad_rate_by_funding_level.png', bbox_inches='tight')
    plt.close()


def main():
    scorecard = load_scorecard("data/college_scorecard_clean.csv")
    census = load_census("data/census_median_income.csv")
    sheeo = load_sheeo("data/SHEEO_FY2024.csv")

    print('Dataset Shapes:')
    print(f'  Scorecard: {scorecard.shape}')
    print(f'  Census: {census.shape}')
    print(f'  SHEEO: {sheeo.shape}')

    plot_bar_by_control(scorecard)
    plot_earnings_by_control(scorecard)
    plot_grad_rate_distribution(scorecard)

    report_missing(scorecard, 'Scorecard')
    report_missing(census, 'Census')
    report_missing(sheeo, 'SHEEO')

    print('\nSummary Statistics:')
    seven_number_summary(scorecard['C150_4'], 'Graduation Rate (C150_4)')
    seven_number_summary(
        scorecard['MD_EARN_WNE_P10'], 'Earnings (MD_EARN_WNE_P10)'
    )
    categorical_summary(scorecard['CONTROL'], 'College Type (CONTROL)')
    seven_number_summary(census['MedianIncome'], 'State Median Income')
    seven_number_summary(sheeo['FundingPerStudent'], 'Funding Per Student')

    merged_scorecard_census = merge_scorecard_census(scorecard, census)
    plot_grad_rate_by_income_level(merged_scorecard_census)
    plot_income_vs_grad_rate(merged_scorecard_census)

    merged_scorecard_sheeo = merge_scorecard_sheeo(scorecard, sheeo)
    plot_funding_vs_grad_rate(merged_scorecard_sheeo)
    plot_grad_rate_by_funding_level(merged_scorecard_sheeo)

    print('\nMerged Dataset Shapes:')
    print(f'  Scorecard + Census: {merged_scorecard_census.shape}')
    print(f'  Scorecard + SHEEO: {merged_scorecard_sheeo.shape}')

    report_missing(merged_scorecard_census, 'Scorecard + Census')
    report_missing(merged_scorecard_sheeo, 'Scorecard + SHEEO')

    categorical_summary(merged_scorecard_census['IncomeLevel'],
                        'Income Level (Scorecard + Census)')
    verify_test_assumptions(scorecard, merged_scorecard_census, 
                            merged_scorecard_sheeo)


if __name__ == "__main__":
    main()
