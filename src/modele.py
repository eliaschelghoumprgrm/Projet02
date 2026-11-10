from __future__ import annotations

import functools
import json
import os
import time
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from exceptions import FichierIntrouvableError, ModeleNonEntraineError
from variables import Variable, VariableCategorielle, VariableNumerique

NOM_MODELE = "modele.joblib"
NOM_METADONNEES = "metadonnees.json"


def chronometrer(fonction: Callable) -> Callable:

    """Décorateur : affiche le temps d'exécution de la fonction décorée."""

    @functools.wraps(fonction)
    def enveloppe(*args: Any, **kwargs: Any) -> Any:

        debut = time.perf_counter()
        resultat = fonction(*args, **kwargs)

        print(f"{fonction.__name__} : {time.perf_counter() - debut:.2f} s")
        return resultat
    
    return enveloppe


def rmse(y_vrai: np.ndarray, y_predit: np.ndarray) -> float:

    """Racine de l'erreur quadratique moyenne, calculée avec NumPy."""

    ecarts = np.asarray(y_vrai, dtype=float) - np.asarray(y_predit, dtype=float)
    return float(np.sqrt(np.mean(ecarts ** 2)))


def r2(y_vrai: np.ndarray, y_predit: np.ndarray) -> float:

    """Coefficient de détermination R², calculé avec NumPy."""

    y_vrai = np.asarray(y_vrai, dtype=float)
    residus = np.sum((y_vrai - np.asarray(y_predit, dtype=float)) ** 2)
    total = np.sum((y_vrai - y_vrai.mean()) ** 2)
    return float(1 - residus / total)


class ModeleNatation:

    """Forêt aléatoire qui estime le temps (en secondes) d'une course olympique.
    Composition : le modèle contient la liste de ses variables d'entrée.
    """

    def __init__(self, variables: list[Variable], nb_arbres: int = 100) -> None:
        
        self._variables = list(variables)
        self._nb_arbres = nb_arbres
        self._pipeline: Pipeline | None = None
        self.metriques: dict[str, float] = {}


    @property
    def variables(self) -> list[Variable]:
        """Copie de la liste des variables d'entrée."""
        return list(self._variables)


    @property
    def colonnes(self) -> list[str]:
        """Noms des colonnes d'entrée, dans l'ordre."""
        return [v.nom for v in self._variables]


    @property
    def est_entraine(self) -> bool:
        """Vrai si le modèle peut faire des prédictions."""
        return self._pipeline is not None



    def _creer_pipeline(self) -> Pipeline:

        """Encodage one-hot des catégories, puis forêt aléatoire."""

        categorielles = [v.nom for v in self._variables if isinstance(v, VariableCategorielle)]
        numeriques = [v.nom for v in self._variables if isinstance(v, VariableNumerique)]

        preparation = ColumnTransformer([
            ("categories", OneHotEncoder(handle_unknown="ignore"), categorielles),
            ("nombres", "passthrough", numeriques),
        ])

        foret = RandomForestRegressor(n_estimators=self._nb_arbres, min_samples_leaf=2, random_state=42, n_jobs=-1)
        return Pipeline([("preparation", preparation), ("foret", foret)])



    @chronometrer
    def entrainer(self, X: pd.DataFrame, y: pd.Series) -> None:

        """Entraîne le modèle sur les données d'entraînement."""

        self._pipeline = self._creer_pipeline()
        self._pipeline.fit(X[self.colonnes], y)



    def predire_lot(self, X: pd.DataFrame) -> np.ndarray:
       
        """Prédit les temps de plusieurs lignes.
        Lève ModeleNonEntraineError si le modèle n'est pas entraîné.
        """

        if self._pipeline is None:
            raise ModeleNonEntraineError("Le modèle n'est pas entraîné.")
        return self._pipeline.predict(X[self.colonnes])



    def valider_saisies(self, saisies: dict[str, str]) -> dict[str, Any]:

        """Valide chaque saisie avec sa variable.
        Lève DonneeInvalideError au premier champ invalide.
        """

        return {v.nom: v.valider(saisies.get(v.nom, "")) for v in self._variables}



    def predire(self, saisies: dict[str, str]) -> float:

        """Valide les saisies de l'utilisateur et renvoie le temps prédit en secondes."""

        valeurs = self.valider_saisies(saisies)
        return float(self.predire_lot(pd.DataFrame([valeurs]))[0])



    def evaluer(self, X_test: pd.DataFrame, y_test: pd.Series, y_train: pd.Series) -> dict[str, float]:
       
        """Calcule RMSE et R² sur le test, et la RMSE de la référence (moyenne d'entraînement)."""

        predictions = self.predire_lot(X_test)
        reference = np.full(len(y_test), float(np.mean(y_train)))

        self.metriques = {
            "rmse": rmse(y_test, predictions),
            "r2": r2(y_test, predictions),
            "rmse_reference": rmse(y_test, reference),
            "r2_reference": r2(y_test, reference),
        }

        return self.metriques



    def importances(self) -> pd.Series:

        """Importance de chaque variable d'entrée (somme des colonnes one-hot)."""

        if self._pipeline is None:
            raise ModeleNonEntraineError("Le modèle n'est pas entraîné.")
        
        noms = self._pipeline.named_steps["preparation"].get_feature_names_out()
        valeurs = self._pipeline.named_steps["foret"].feature_importances_
        par_variable = {v.nom: 0.0 for v in self._variables}

        for nom_colonne, valeur in zip(noms, valeurs):
            nom_variable = nom_colonne.split("__", 1)[1]

            for v in self._variables:
                if nom_variable == v.nom or nom_variable.startswith(v.nom + "_"):
                    par_variable[v.nom] += float(valeur)
                    break

        return pd.Series(par_variable).sort_values()



    def sauvegarder(self, dossier: str) -> None:

        """Enregistre le modèle (joblib) et ses métadonnées (JSON) dans le dossier."""

        if self._pipeline is None:
            raise ModeleNonEntraineError("Impossible de sauvegarder un modèle non entraîné.")
        
        os.makedirs(dossier, exist_ok=True)
        joblib.dump(self._pipeline, os.path.join(dossier, NOM_MODELE))

        metadonnees = {
            "cible": "temps_s",
            "colonnes": self.colonnes,
            "variables": [v.vers_dict() for v in self._variables],
            "metriques": self.metriques,
        }

        with open(os.path.join(dossier, NOM_METADONNEES), "w", encoding="utf-8") as fichier:
            json.dump(metadonnees, fichier, ensure_ascii=False, indent=2)



    @classmethod
    def charger(cls, dossier: str) -> ModeleNatation:

        """Recharge un modèle sauvegardé.
        Lève FichierIntrouvableError s'il manque le modèle ou les métadonnées.
        """

        chemin_modele = os.path.join(dossier, NOM_MODELE)
        chemin_meta = os.path.join(dossier, NOM_METADONNEES)

        for chemin in (chemin_modele, chemin_meta):
            if not os.path.exists(chemin):
                raise FichierIntrouvableError(f"Fichier introuvable : {chemin}")
            
        with open(chemin_meta, encoding="utf-8") as fichier:
            metadonnees = json.load(fichier)

        modele = cls([Variable.depuis_dict(d) for d in metadonnees["variables"]])
        modele._pipeline = joblib.load(chemin_modele)
        modele.metriques = metadonnees.get("metriques", {})

        return modele


    def __repr__(self) -> str:
        etat = "entraîné" if self.est_entraine else "non entraîné"
        return f"ModeleNatation({len(self._variables)} variables, {etat})"
