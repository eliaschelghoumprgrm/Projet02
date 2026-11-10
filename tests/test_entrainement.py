"""Test de bout en bout : vraies données -> modèle sauvegardé et graphiques."""
import os

from donnees import JeuDeDonnees
from entrainement import entrainer_et_sauvegarder
from modele import ModeleNatation


def test_chaine_complete(tmp_path):

    jeu = JeuDeDonnees().charger()
    modeles, figures = os.path.join(tmp_path, "m"), os.path.join(tmp_path, "f")
    modele = entrainer_et_sauvegarder(jeu, modeles, figures)

    assert modele.metriques["rmse"] < modele.metriques["rmse_reference"]
    assert os.path.exists(os.path.join(figures, "exploration.png"))
    assert os.path.exists(os.path.join(figures, "predit_contre_reel.png"))

    recharge = ModeleNatation.charger(modeles)
    saisie = {"annee": "2020", "distance_m": "100", "nage": "Libre",
              "genre": "Hommes", "relais": "Non", "equipe": "USA"}
    
    assert 40 < recharge.predire(saisie) < 60
