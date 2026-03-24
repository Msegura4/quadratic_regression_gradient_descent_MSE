# Quadratic Regression by Gradient Descent (MSE)

Implémentation d'une régression quadratique entraînée par descente de gradient.

## Fichiers
- `main_linear.py` : régression linéaire
- `main_quadratic.py` : régression quadratique

## Installation
```bash
pip install -r requirements.txt
```

## Utilisation
Avant de lancer, modifier le chemin absolu du CSV dans les deux fichiers :
```python
df = pd.read_csv('/votre/chemin/vers/prix_maisons.csv')
```

## Notes
- La régression quadratique nécessite entre 44 000 et 65 000 epochs pour converger.
- Le learning rate utilisé est `epsilon = 0.0001`.
