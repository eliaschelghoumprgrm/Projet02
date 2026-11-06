"""Variables d'entrée du modèle : chacune sait valider la saisie de l'utilisateur.

Héritage et polymorphisme : Variable (abstraite) -> VariableNumerique et
VariableCategorielle, qui redéfinissent toutes les deux valider().
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from exceptions import DonneeInvalideError


class Variable(ABC):
    """Variable d'entrée du modèle (classe abstraite)."""

    def __init__(self, nom: str, libelle: str) -> None:
        """Reçoit le nom de la colonne et le libellé affiché à l'utilisateur."""
        self._nom = nom
        self._libelle = libelle

    @property
    def nom(self) -> str:
        """Nom de la colonne dans les données."""
        return self._nom

    @property
    def libelle(self) -> str:
        """Texte affiché dans l'interface."""
        return self._libelle

    @abstractmethod
    def valider(self, saisie: str) -> Any:
        """Convertit la saisie (texte) en valeur utilisable par le modèle.

        Lève DonneeInvalideError si la saisie est vide ou invalide.
        """

    @abstractmethod
    def vers_dict(self) -> dict:
        """Renvoie la description de la variable pour le fichier JSON."""

    def _verifier_non_vide(self, saisie: str) -> str:
        """Renvoie la saisie sans espaces ; lève DonneeInvalideError si elle est vide."""
        texte = str(saisie).strip()
        if texte == "":
            raise DonneeInvalideError(f"Le champ « {self._libelle} » est vide.")
        return texte

    @staticmethod
    def depuis_dict(description: dict) -> "Variable":
        """Recrée une variable à partir de sa description JSON.

        Lève DonneeInvalideError si le type est inconnu.
        """
        if description["type"] == "numerique":
            return VariableNumerique(description["nom"], description["libelle"],
                                     description["min"], description["max"])
        if description["type"] == "categorielle":
            return VariableCategorielle(description["nom"], description["libelle"],
                                        description["valeurs"])
        raise DonneeInvalideError(f"Type de variable inconnu : {description['type']}")

    def __str__(self) -> str:
        """Texte lisible, par exemple pour l'affichage dans la console."""
        return f"{self._libelle} ({self._nom})"

    def __eq__(self, autre: object) -> bool:
        """Deux variables sont égales si elles ont la même description."""
        if not isinstance(autre, Variable):
            return NotImplemented
        return self.vers_dict() == autre.vers_dict()


class VariableNumerique(Variable):
    """Variable numérique bornée par un intervalle [minimum, maximum]."""

    def __init__(self, nom: str, libelle: str, minimum: float, maximum: float) -> None:
        """Reçoit le nom, le libellé et l'intervalle permis."""
        super().__init__(nom, libelle)
        self._min = minimum
        self._max = maximum

    @property
    def minimum(self) -> float:
        """Plus petite valeur permise."""
        return self._min

    @property
    def maximum(self) -> float:
        """Plus grande valeur permise."""
        return self._max

    def valider(self, saisie: str) -> float:
        """Renvoie la saisie convertie en nombre.

        Lève DonneeInvalideError si elle est vide, non numérique ou hors intervalle.
        """
        texte = self._verifier_non_vide(saisie).replace(",", ".")
        try:
            valeur = float(texte)
        except ValueError:
            raise DonneeInvalideError(
                f"« {self._libelle} » doit être un nombre (reçu : « {texte} »).") from None
        if not self._min <= valeur <= self._max:
            raise DonneeInvalideError(
                f"« {self._libelle} » doit être entre {self._min:g} et {self._max:g}.")
        return valeur

    def vers_dict(self) -> dict:
        """Description JSON de la variable."""
        return {"type": "numerique", "nom": self._nom, "libelle": self._libelle,
                "min": self._min, "max": self._max}

    def __repr__(self) -> str:
        """Représentation pour le débogage."""
        return f"VariableNumerique({self._nom!r}, {self._min}, {self._max})"


class VariableCategorielle(Variable):
    """Variable qui prend une valeur parmi une liste fixe."""

    def __init__(self, nom: str, libelle: str, valeurs: list[str]) -> None:
        """Reçoit le nom, le libellé et la liste des valeurs possibles."""
        super().__init__(nom, libelle)
        self._valeurs = list(valeurs)

    @property
    def valeurs(self) -> list[str]:
        """Copie de la liste des valeurs possibles."""
        return list(self._valeurs)

    def valider(self, saisie: str) -> str:
        """Renvoie la saisie si elle fait partie des valeurs possibles.

        Lève DonneeInvalideError sinon.
        """
        texte = self._verifier_non_vide(saisie)
        if texte not in self._valeurs:
            raise DonneeInvalideError(
                f"« {texte} » n'est pas une valeur permise pour « {self._libelle} ».")
        return texte

    def vers_dict(self) -> dict:
        """Description JSON de la variable."""
        return {"type": "categorielle", "nom": self._nom, "libelle": self._libelle,
                "valeurs": self._valeurs}

    def __repr__(self) -> str:
        """Représentation pour le débogage."""
        return f"VariableCategorielle({self._nom!r}, {len(self._valeurs)} valeurs)"
