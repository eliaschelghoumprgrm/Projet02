"""Premier test pour vérifier que la CI fonctionne."""
import numpy as np


def test_numpy_fonctionne() -> None:
    """Vérifie qu'un calcul NumPy simple donne le bon résultat."""
    assert np.mean([1, 2, 3]) == 2
