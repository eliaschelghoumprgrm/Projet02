"""Tests de la lecture et du nettoyage des données."""
import os

import pytest

from donnees import (CIBLE, JeuDeDonnees, convertir_distance, convertir_temps,
                     formater_temps, lire_csv_par_blocs, nettoyer)
from exceptions import DonneeInvalideError, FichierIntrouvableError


@pytest.mark.parametrize("texte, attendu", [
    ("51.98", 51.98),                 # secondes seules
    ("1:53.27", 113.27),              # minutes:secondes
    ("00:01:04.930000", 64.93),       # heures:minutes:secondes
    ("22:36.4est", 1356.4),           # temps estimé
])
def test_convertir_temps_formats(texte, attendu):
    assert convertir_temps(texte) == pytest.approx(attendu)


@pytest.mark.parametrize("texte", ["Disqualified", "Did not start", "", "0", "1:2:3:4"])
def test_convertir_temps_invalide(texte):
    with pytest.raises(DonneeInvalideError):
        convertir_temps(texte)


def test_convertir_distance():
    assert convertir_distance("100m") == 100
    assert convertir_distance("4x200") == 800
    with pytest.raises(DonneeInvalideError):
        convertir_distance("cent metres")


def test_formater_temps():
    assert formater_temps(47.0) == "47.00 s"
    assert formater_temps(125.4) == "2:05.40"


def test_nettoyer(brut):
    df = nettoyer(brut)
    # DQ, vide et 1 s au 50 m (aberrant) retirés
    assert len(df) == 5
    assert "Athlete" not in df.columns and "Rank" not in df.columns
    assert set(df["equipe"]) == {"Autre"}          # chaque pays < 10 lignes ici
    relais = df[df["relais"] == "Oui"].iloc[0]
    assert relais["distance_m"] == 800
    assert relais[CIBLE] == pytest.approx(419.7)
    assert df["nage"].iloc[1] == "Dos"


def test_lire_csv_par_blocs(tmp_path, brut):
    chemin = os.path.join(tmp_path, "petit.csv")
    brut.to_csv(chemin, index=False)
    blocs = list(lire_csv_par_blocs(chemin, taille_bloc=3))
    assert [len(b) for b in blocs] == [3, 3, 2]


def test_fichier_introuvable(tmp_path):
    with pytest.raises(FichierIntrouvableError):
        next(lire_csv_par_blocs(os.path.join(tmp_path, "absent.csv")))


def test_jeu_de_donnees_vrai_fichier():
    jeu = JeuDeDonnees().charger()
    assert 500 <= len(jeu) <= 50_000
    assert len(jeu.X.columns) >= 5
    assert jeu.y.min() > 0
    noms = [v.nom for v in jeu.construire_variables()]
    assert noms == list(jeu.X.columns)


def test_jeu_non_charge():
    with pytest.raises(DonneeInvalideError):
        JeuDeDonnees().donnees
