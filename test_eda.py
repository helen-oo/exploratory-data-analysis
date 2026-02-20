"""
Jenny Hoang, Helen Jiang
CSE 163
Final Project - Testing

This program tests the key functions in eda.py using small,
hand-crafted DataFrames with known values so we can verify
that our results are correct.
"""

import pandas as pd
from eda import (
    merge_scorecard_census,
    merge_scorecard_sheeo,
)


def make_scorecard() -> pd.DataFrame:
    """
    Returns a small fake Scorecard DataFrame with 6 colleges
    for testing purposes.
    """
    return pd.DataFrame({
        'INSTNM': ['CollegeA', 'CollegeB', 'CollegeC',
                   'CollegeD', 'CollegeE', 'CollegeF'],
        'STABBR': ['WA', 'WA', 'CA', 'CA', 'TX', 'TX'],
        'CONTROL': ['Public', 'Private Nonprofit',
                    'Public', 'Private Nonprofit',
                    'Public', 'Private Nonprofit'],
        'C150_4': [0.5, 0.7, 0.4, 0.6, 0.3, 0.8],
        'MD_EARN_WNE_P10': [50000, 60000, 45000, 55000, 40000, 70000],
        'STATE_NAME': ['Washington', 'Washington',
                       'California', 'California',
                       'Texas', 'Texas'],
    })


def make_census() -> pd.DataFrame:
    """
    Returns a small fake Census DataFrame with 3 states
    for testing purposes.
    """
    return pd.DataFrame({
        'State': ['Washington', 'California', 'Texas'],
        'MedianIncome': [90000, 80000, 65000],
    })


def make_sheeo() -> pd.DataFrame:
    """
    Returns a small fake SHEEO DataFrame with 3 states
    and pre-computed FundingPerStudent for testing purposes.
    """
    return pd.DataFrame({
        'State': ['Washington', 'California', 'Texas'],
        'Education Appropriations': [1000000, 2000000, 1500000],
        'Net FTE Enrollment': [10000, 20000, 15000],
        'FundingPerStudent': [100.0, 100.0, 100.0],
    })


def test_merge_scorecard_census() -> None:
    """
    Tests that merge_scorecard_census correctly merges and
    adds an IncomeLevel column.
    """
    scorecard = make_scorecard()
    census = make_census()
    merged = merge_scorecard_census(scorecard, census)

    # All 3 states matched so all 6 rows should be retained
    assert len(merged) == 6, (
        f'Expected 6 rows after merge, got {len(merged)}'
    )

    # IncomeLevel column should exist
    assert 'IncomeLevel' in merged.columns, (
        'IncomeLevel column missing after merge'
    )

    # Washington has highest income so should be High Income
    wa_level = merged[merged['STATE_NAME'] == 'Washington'][
        'IncomeLevel'].iloc[0]
    assert str(wa_level) == 'High Income', (
        f'Expected Washington to be High Income, got {wa_level}'
    )

    # Texas has lowest income so should be Low Income
    tx_level = merged[merged['STATE_NAME'] == 'Texas'][
        'IncomeLevel'].iloc[0]
    assert str(tx_level) == 'Low Income', (
        f'Expected Texas to be Low Income, got {tx_level}'
    )

    print('test_merge_scorecard_census passed!')


def test_merge_scorecard_sheeo() -> None:
    """
    Tests that merge_scorecard_sheeo correctly filters to public
    colleges only and merges with SHEEO data.
    """
    scorecard = make_scorecard()
    sheeo = make_sheeo()
    merged = merge_scorecard_sheeo(scorecard, sheeo)

    # Should have one row per state
    assert len(merged) == 3, (
        f'Expected 3 rows (one per state), got {len(merged)}'
    )

    # Should have AvgGradRate column
    assert 'AvgGradRate' in merged.columns, (
        'AvgGradRate column missing after merge'
    )

    # Avg should be 0.5
    wa_row = merged[merged['STATE_NAME'] == 'Washington']
    assert abs(wa_row['AvgGradRate'].iloc[0] - 0.5) < 0.001, (
        f'Expected WA AvgGradRate=0.5, got {wa_row["AvgGradRate"].iloc[0]}'
    )

    # FundingPerStudent column should be present from SHEEO
    assert 'FundingPerStudent' in merged.columns, (
        'FundingPerStudent column missing after merge'
    )

    print('test_merge_scorecard_sheeo passed!')


def test_merge_drops_unmatched_states() -> None:
    scorecard = make_scorecard()

    # Census only has Washington and California, not Texas
    census_partial = pd.DataFrame({
        'State': ['Washington', 'California'],
        'MedianIncome': [90000, 80000],
    })

    merged = merge_scorecard_census(scorecard, census_partial)

    # Only Washington and California colleges should remain
    assert len(merged) == 4, (
        f'Expected 4 rows after partial merge, got {len(merged)}'
    )
    assert 'Texas' not in merged['STATE_NAME'].values, (
        'Texas rows should have been dropped'
    )

    print('test_merge_drops_unmatched_states passed!')


def test_income_level_assignment() -> None:
    """
    Tests that IncomeLevel categories are assigned correctly.
    Colleges in the highest income state should be High Income
    and colleges in the lowest income state should be Low Income.
    """
    scorecard = make_scorecard()
    census = make_census()
    merged = merge_scorecard_census(scorecard, census)

    # Washington has highest income (90000) = High Income
    wa_rows = merged[merged['STATE_NAME'] == 'Washington']
    assert all(wa_rows['IncomeLevel'] == 'High Income'), (
        'All Washington colleges should be High Income'
    )

    # Texas has lowest income (65000) = Low Income
    tx_rows = merged[merged['STATE_NAME'] == 'Texas']
    assert all(tx_rows['IncomeLevel'] == 'Low Income'), (
        'All Texas colleges should be Low Income'
    )

    # No null income levels should exist
    assert merged['IncomeLevel'].isna().sum() == 0, (
        'No college should have a missing IncomeLevel'
    )

    print('test_income_level_assignment passed!')


def main():
    test_merge_scorecard_census()
    test_merge_scorecard_sheeo()
    test_merge_drops_unmatched_states()
    test_income_level_assignment()
    print('\nAll tests passed!')


if __name__ == "__main__":
    main()
