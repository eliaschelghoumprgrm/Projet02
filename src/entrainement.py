"""Chaîne complète : données -> modèle entraîné, évalué et sauvegardé."""
from __future__ import annotations

import os

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from donnees import COLONNES_CATEGORIELLES, COLONNES_NUMERIQUES, RACINE, JeuDeDonnees
from graphiques import figure_exploration, figure_predit_contre_reel
from modele import ModeleNatation, rmse

DOSSIER_MODELES = os.path.join(RACINE, "modeles")
DOSSIER_FIGURES = os.path.join(RACINE, "figures")
GRAINE = 42


def rmse_regression_lineaire(X_train, y_train, X_test, y_test) -> float:

    """Second modèle, pour comparer : régression linéaire (RMSE sur le test)."""

    pipeline = Pipeline([
        ("preparation", ColumnTransformer([
            ("categories", OneHotEncoder(handle_unknown="ignore"), COLONNES_CATEGORIELLES),
            ("nombres", "passthrough", COLONNES_NUMERIQUES),
        ])),
        ("lineaire", LinearRegression()),
    ])
    
    pipeline.fit(X_train, y_train)

    return rmse(y_test, pipeline.predict(X_test))


def entrainer_et_sauvegarder(jeu: JeuDeDonnees, dossier_modeles: str = DOSSIER_MODELES, dossier_figures: str = DOSSIER_FIGURES) -> ModeleNatation:
   
    """Sépare les données, entraîne, évalue, sauvegarde le modèle et les graphiques."""

    os.makedirs(dossier_figures, exist_ok=True)
    figure_exploration(jeu.donnees).savefig(os.path.join(dossier_figures, "exploration.png"), dpi=120)

    X_train, X_test, y_train, y_test = train_test_split(jeu.X, jeu.y, test_size=0.2, random_state=GRAINE)

    modele = ModeleNatation(jeu.construire_variables())

    modele.entrainer(X_train, y_train)
    modele.evaluer(X_test, y_test, y_train)
    modele.metriques["rmse_regression_lineaire"] = rmse_regression_lineaire(X_train, y_train, X_test, y_test)
    modele.metriques["nb_lignes_entrainement"] = len(X_train)
    modele.metriques["nb_lignes_test"] = len(X_test)
    modele.sauvegarder(dossier_modeles)

    figure_predit_contre_reel(y_test.to_numpy(), modele.predire_lot(X_test)).savefig(
        os.path.join(dossier_figures, "predit_contre_reel.png"), dpi=120)
    
    return modele




def afficher_resultats(jeu: JeuDeDonnees, modele: ModeleNatation) -> None:

    """Affiche un résumé lisible des résultats dans la console."""

    m = modele.metriques

    print(f"Données : {len(jeu)} courses gardées, {jeu.nb_lignes_retirees} retirées au nettoyage")
    print(f"Forêt aléatoire        : RMSE = {m['rmse']:.2f} s, R² = {m['r2']:.3f}")
    print(f"Référence (moyenne)    : RMSE = {m['rmse_reference']:.2f} s, R² = {m['r2_reference']:.3f}")
    print(f"Régression linéaire    : RMSE = {m['rmse_regression_lineaire']:.2f} s")
    print("Importance des variables :")

    for nom, valeur in modele.importances().sort_values(ascending=False).items():
        print(f"  {nom:<12} {valeur:.3f}")
