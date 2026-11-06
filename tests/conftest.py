"""Données communes aux tests : un petit jeu au format du vrai CSV."""
import pandas as pd
import pytest


@pytest.fixture
def brut() -> pd.DataFrame:
    """Huit lignes brutes, dont trois à retirer au nettoyage."""
    return pd.DataFrame({
        "Location": ["Tokyo"] * 8,
        "Year": ["2020", "2016", "2012", "2008", "2004", "2000", "1996", "1992"],
        "Distance (in meters)": ["100m", "100m", "4x200", "200m", "100m", "100m", "50m", "100m"],
        "Stroke": ["Freestyle", "Backstroke", "Freestyle", "Butterfly",
                   "Freestyle", "Freestyle", "Freestyle", "Freestyle"],
        "Relay?": ["0", "0", "1", "0", "0", "0", "0", "0"],
        "Gender": ["Men", "Women", "Men", "Women", "Men", "Men", "Men", "Men"],
        "Team": ["USA", "NL", "NLD", "NED", "USA", "USA", "USA", "Unified Team "],
        "Athlete": ["A", "B", "C", "D", "E", "F", "G", "H"],
        "Results": ["47.02", "00:00:58.450000", "6:59.70", "2:04.06",
                    "Disqualified", "", "1.00", "48.300"],
        "Rank": ["1", "1", "1", "1", "0", "0", "4", "1"],
    })
