"""Exceptions personnalisées du projet."""


class ProjetErreur(Exception):
    """Classe parente de toutes les erreurs propres au projet."""


class DonneeInvalideError(ProjetErreur):
    """Levée quand une donnée (CSV ou saisie de l'utilisateur) est invalide."""


class FichierIntrouvableError(ProjetErreur):
    """Levée quand un fichier nécessaire (CSV, modèle, métadonnées) n'existe pas."""


class ModeleNonEntraineError(ProjetErreur):
    """Levée quand on demande une prédiction à un modèle qui n'est pas entraîné."""
