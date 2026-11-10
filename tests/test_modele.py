"""Tests du modèle : métriques NumPy, entraînement, prédiction, sauvegarde."""
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import mean_squared_error, r2_score

from exceptions import DonneeInvalideError, FichierIntrouvableError, ModeleNonEntraineError
from modele import ModeleNatation, r2, rmse
from variables import VariableCategorielle, VariableNumerique


def test_rmse_comme_sklearn():
    
    y_vrai = np.array([50.0, 60.0, 120.0, 240.0])
    y_predit = np.array([52.0, 58.0, 125.0, 230.0])

    assert rmse(y_vrai, y_predit) == pytest.approx(np.sqrt(mean_squared_error(y_vrai, y_predit)))
    assert rmse(y_vrai, y_vrai) == pytest.approx(0.0)


def test_r2_comme_sklearn():

    y_vrai = np.array([50.0, 60.0, 120.0, 240.0])
    y_predit = np.array([52.0, 58.0, 125.0, 230.0])

    assert r2(y_vrai, y_predit) == pytest.approx(r2_score(y_vrai, y_predit))


@pytest.fixture
def donnees():
    """Petit jeu : le temps dépend surtout de la distance."""
    rng = np.random.default_rng(0)
    n = 200
    distance = rng.choice([100, 200, 400], n)
    nage = rng.choice(["Libre", "Dos"], n)

    X = pd.DataFrame({"distance_m": distance, "nage": nage})
    y = pd.Series(distance * 0.55 + np.where(nage == "Dos", 5, 0) + rng.normal(0, 1, n))

    return X, y


@pytest.fixture
def modele():
    return ModeleNatation([VariableNumerique("distance_m", "Distance", 50, 1500), VariableCategorielle("nage", "Nage", ["Dos", "Libre"])], nb_arbres=20)

def test_meilleur_que_reference(modele, donnees):
    X, y = donnees

    modele.entrainer(X[:150], y[:150])

    m = modele.evaluer(X[150:], y[150:], y[:150])

    assert m["rmse"] < m["rmse_reference"]
    assert m["r2"] > 0.9


def test_prediction_saisie(modele, donnees):

    modele.entrainer(*donnees)

    assert modele.predire({"distance_m": "200", "nage": "Libre"}) == pytest.approx(110, abs=5)

    with pytest.raises(DonneeInvalideError):
        modele.predire({"distance_m": "abc", "nage": "Libre"})


def test_non_entraine(modele):

    with pytest.raises(ModeleNonEntraineError):
        modele.predire({"distance_m": "100", "nage": "Dos"})


def test_sauvegarde_et_chargement(tmp_path, modele, donnees):

    modele.entrainer(*donnees)
    modele.sauvegarder(str(tmp_path))

    recharge = ModeleNatation.charger(str(tmp_path))
    saisie = {"distance_m": "400", "nage": "Dos"}

    assert recharge.predire(saisie) == pytest.approx(modele.predire(saisie))
    assert recharge.variables == modele.variables


def test_chargement_dossier_vide(tmp_path):

    with pytest.raises(FichierIntrouvableError):
        ModeleNatation.charger(str(tmp_path))
