"""Lecture et nettoyage du jeu de données de natation olympique."""
from __future__ import annotations

import os
from typing import Iterator

import numpy as np
import pandas as pd

from exceptions import DonneeInvalideError, FichierIntrouvableError
from variables import Variable, VariableCategorielle, VariableNumerique

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHEMIN_CSV = os.path.join(RACINE, "donnees", "Olympic_Swimming_Results_1912to2020.csv")

CIBLE = "temps_s"
COLONNES_NUMERIQUES = ["annee", "distance_m"]
COLONNES_CATEGORIELLES = ["nage", "genre", "relais", "equipe"]
COLONNES_ENTREE = COLONNES_NUMERIQUES + COLONNES_CATEGORIELLES

LIBELLES = {
    "annee": "Année des Jeux",
    "distance_m": "Distance totale (m)",
    "nage": "Nage",
    "genre": "Catégorie",
    "relais": "Relais",
    "equipe": "Pays",
}

TRADUCTION_NAGES = {
    "Freestyle": "Libre",
    "Backstroke": "Dos",
    "Breaststroke": "Brasse",
    "Butterfly": "Papillon",
    "Individual medley": "4 nages individuel",
    "Medley": "4 nages (relais)",
}
TRADUCTION_GENRES = {"Men": "Hommes", "Women": "Femmes"}

# Même pays écrit de plusieurs façons dans le fichier.
CODES_EQUIVALENTS = {"NL": "NED", "NLD": "NED", "NZ": "NZL", "ZAF": "RSA",
                     "SUN": "URS", "Unified Team": "EUN"}

MIN_LIGNES_PAR_EQUIPE = 10          # en dessous : regroupé dans « Autre »
VITESSE_MIN, VITESSE_MAX = 0.7, 2.5  # m/s ; en dehors : valeur aberrante


def lire_csv_par_blocs(chemin: str, taille_bloc: int = 1000) -> Iterator[pd.DataFrame]:
    """Générateur : renvoie le CSV morceau par morceau (taille_bloc lignes).

    Lève FichierIntrouvableError si le fichier n'existe pas.
    """
    if not os.path.exists(chemin):
        raise FichierIntrouvableError(f"Fichier introuvable : {chemin}")
    with open(chemin, encoding="utf-8-sig") as fichier:
        for bloc in pd.read_csv(fichier, chunksize=taille_bloc, dtype=str):
            yield bloc


def charger_donnees_brutes(chemin: str = CHEMIN_CSV) -> pd.DataFrame:
    """Lit tout le CSV (bloc par bloc) et renvoie un DataFrame brut."""
    return pd.concat(list(lire_csv_par_blocs(chemin)), ignore_index=True)


def convertir_temps(texte: str) -> float:
    """Convertit un résultat (« 51.98 », « 1:53.27 », « 00:01:04.93 ») en secondes.

    Lève DonneeInvalideError si le texte n'est pas un temps (ex. « Disqualified »).
    """
    if not isinstance(texte, str) or texte.strip() == "":
        raise DonneeInvalideError("Résultat vide.")
    nettoye = texte.strip().removesuffix("est")
    try:
        parties = [float(p) for p in nettoye.split(":")]
    except ValueError:
        raise DonneeInvalideError(f"Résultat non numérique : {texte!r}") from None
    if len(parties) > 3:
        raise DonneeInvalideError(f"Format de temps inconnu : {texte!r}")
    secondes = 0.0
    for partie in parties:          # heures -> minutes -> secondes
        secondes = secondes * 60 + partie
    if secondes <= 0:
        raise DonneeInvalideError(f"Temps nul ou négatif : {texte!r}")
    return secondes


def convertir_distance(texte: str) -> int:
    """Convertit « 100m » en 100 et un relais « 4x200 » en 800 (distance totale).

    Lève DonneeInvalideError si le format est inconnu.
    """
    nettoye = str(texte).strip().lower().removesuffix("m")
    try:
        if "x" in nettoye:
            nageurs, distance = nettoye.split("x")
            return int(nageurs) * int(distance)
        return int(nettoye)
    except ValueError:
        raise DonneeInvalideError(f"Distance invalide : {texte!r}") from None


def formater_temps(secondes: float) -> str:
    """Affiche un temps en secondes sous la forme « m:ss.cc » (ex. 125.4 -> « 2:05.40 »)."""
    minutes, reste = divmod(round(secondes, 2), 60)
    if minutes == 0:
        return f"{reste:.2f} s"
    return f"{int(minutes)}:{reste:05.2f}"


def _temps_ou_nan(texte: str) -> float:
    """Comme convertir_temps, mais renvoie NaN au lieu de lever une exception."""
    try:
        return convertir_temps(texte)
    except DonneeInvalideError:
        return np.nan


def nettoyer(brut: pd.DataFrame) -> pd.DataFrame:
    """Renvoie un DataFrame propre, prêt pour le modèle.

    - retire Athlete (noms : données personnelles), Rank (connu après la course)
      et Location (redondante avec l'année) ;
    - convertit les résultats en secondes et retire les lignes sans temps ;
    - convertit les distances en nombres, traduit les catégories en français ;
    - harmonise les codes de pays et regroupe les pays rares dans « Autre » ;
    - retire les temps aberrants (vitesse impossible).
    """
    df = pd.DataFrame({
        "annee": pd.to_numeric(brut["Year"], errors="coerce"),
        "distance_m": brut["Distance (in meters)"].map(convertir_distance),
        "nage": brut["Stroke"].str.strip().map(TRADUCTION_NAGES),
        "genre": brut["Gender"].str.strip().map(TRADUCTION_GENRES),
        "relais": brut["Relay?"].astype(str).str.strip().map({"0": "Non", "1": "Oui"}),
        "equipe": brut["Team"].str.strip().replace(CODES_EQUIVALENTS),
        CIBLE: brut["Results"].map(_temps_ou_nan),
    })
    df = df.dropna().copy()
    df["annee"] = df["annee"].astype(int)

    frequences = df["equipe"].value_counts()
    rares = frequences[frequences < MIN_LIGNES_PAR_EQUIPE].index
    df.loc[df["equipe"].isin(rares), "equipe"] = "Autre"

    vitesse = df["distance_m"].to_numpy() / df[CIBLE].to_numpy()   # calcul vectorisé
    df = df[(vitesse >= VITESSE_MIN) & (vitesse <= VITESSE_MAX)]
    return df.reset_index(drop=True)


class JeuDeDonnees:
    """Jeu de données nettoyé, avec les variables d'entrée qu'il définit."""

    def __init__(self, chemin: str = CHEMIN_CSV) -> None:
        """Reçoit le chemin du CSV ; les données sont lues par charger()."""
        self._chemin = chemin
        self._df: pd.DataFrame | None = None
        self._nb_lignes_brutes = 0

    def charger(self) -> "JeuDeDonnees":
        """Lit et nettoie le CSV ; renvoie self pour pouvoir enchaîner."""
        brut = charger_donnees_brutes(self._chemin)
        self._nb_lignes_brutes = len(brut)
        self._df = nettoyer(brut)
        return self

    @property
    def donnees(self) -> pd.DataFrame:
        """Données nettoyées. Lève DonneeInvalideError si charger() n'a pas été appelé."""
        if self._df is None:
            raise DonneeInvalideError("Les données ne sont pas chargées.")
        return self._df

    @property
    def nb_lignes_retirees(self) -> int:
        """Nombre de lignes retirées par le nettoyage."""
        return self._nb_lignes_brutes - len(self.donnees)

    @property
    def X(self) -> pd.DataFrame:
        """Colonnes d'entrée du modèle."""
        return self.donnees[COLONNES_ENTREE]

    @property
    def y(self) -> pd.Series:
        """Colonne cible (temps en secondes)."""
        return self.donnees[CIBLE]

    def construire_variables(self) -> list[Variable]:
        """Crée une variable par colonne d'entrée, à partir des données."""
        df = self.donnees
        variables: list[Variable] = []
        for nom in COLONNES_NUMERIQUES:
            variables.append(VariableNumerique(nom, LIBELLES[nom],
                                               int(df[nom].min()), int(df[nom].max())))
        for nom in COLONNES_CATEGORIELLES:
            valeurs = sorted(df[nom].unique(), key=lambda v: (v == "Autre", v.lower()))
            variables.append(VariableCategorielle(nom, LIBELLES[nom], valeurs))
        return variables

    def epreuve(self, distance_m: float, nage: str, genre: str, relais: str) -> pd.DataFrame:
        """Renvoie tous les résultats réels d'une épreuve (toutes années confondues)."""
        df = self.donnees
        masque = ((df["distance_m"] == distance_m) & (df["nage"] == nage)
                  & (df["genre"] == genre) & (df["relais"] == relais))
        return df[masque]

    def __len__(self) -> int:
        """Nombre de lignes après nettoyage."""
        return len(self.donnees)

    def __repr__(self) -> str:
        """Représentation pour le débogage."""
        etat = f"{len(self)} lignes" if self._df is not None else "non chargé"
        return f"JeuDeDonnees({os.path.basename(self._chemin)!r}, {etat})"
