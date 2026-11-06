"""Tests des classes de variables (validation des saisies)."""
import pytest

from exceptions import DonneeInvalideError
from variables import Variable, VariableCategorielle, VariableNumerique


@pytest.fixture
def annee():
    return VariableNumerique("annee", "Année", 1912, 2020)


@pytest.fixture
def nage():
    return VariableCategorielle("nage", "Nage", ["Dos", "Libre"])


def test_numerique_valide(annee):
    assert annee.valider(" 2000 ") == pytest.approx(2000)
    assert annee.valider("1912") == pytest.approx(1912)   # borne incluse


@pytest.mark.parametrize("saisie", ["", "   ", "deux mille", "1900", "2021"])
def test_numerique_invalide(annee, saisie):
    with pytest.raises(DonneeInvalideError):
        annee.valider(saisie)


def test_categorielle(nage):
    assert nage.valider("Dos") == "Dos"
    with pytest.raises(DonneeInvalideError):
        nage.valider("Crawl")
    with pytest.raises(DonneeInvalideError):
        nage.valider("")


def test_aller_retour_json(annee, nage):
    assert Variable.depuis_dict(annee.vers_dict()) == annee
    assert Variable.depuis_dict(nage.vers_dict()) == nage
    assert annee != nage
    assert str(annee) == "Année (annee)"
