import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Affichage des dnnées
df = pd.read_csv('/Users/ms/Library/CloudStorage/SynologyDrive-cefedemaura/cours/projet_DIA2/regression_quadratique_descente_gradient/prix_maisons.csv')
print("Aperçu des données :")
print(df.head(11))
print("\n")
print(len(df))


# Standardisation des données : x <- (x - μx) / σx    y <- (y - μy) / σy
house_prices_df = pd.read_csv("prix_maisons.csv")
x_mean, x_std = house_prices_df["surface"].mean(), house_prices_df["surface"].std()
y_mean, y_std = house_prices_df["prix"].mean(), house_prices_df["prix"].std()
house_prices_df["surface"] = (house_prices_df["surface"] - x_mean) / x_std
house_prices_df["prix"] = (house_prices_df["prix"] - y_mean) / y_std


# Visualisation des données
# plt.scatter(house_prices_df["surface"], house_prices_df["prix"], label = 'données')
# plt.legend()
# plt.show()

# QUADRATIC_REGRESSION
# Initialisation
a = np.random.randn()
b = np.random.randn()
c = np.random.randn()
x = house_prices_df["surface"]
y = house_prices_df["prix"]

# Fonction
def quadratic_regression(a, b, c, x):
    return a*x**2 + b*x + c

# MSE
def mse(a, b, c, x, y):
    n = len(x)
    modelqdr = quadratic_regression(a, b, c, x)
    return (1/n) * np.sum((modelqdr - y)**2)

# RMSE
def rmse(a, b, c, x, y):
    return np.sqrt(mse(a, b, c, x, y))



# BACKPORPAGATION_QUADRATIC
# 	Pour créer la fonction nous avons besoin des éléments suivants :

# 		Calcul des gradients (cf. fiche calculs à la main) :
#			- ∂L = (1/n) * Σᵢ₌₁ⁿ (2u * u') avec u = axᵢ² + bxᵢ + c - yᵢ
#			- Par exemple dans notre cas par rapport à a nous avons ∂L/∂a = (2/n) * Σᵢ₌₁ⁿ eᵢ * xᵢ² autrement dit dL_da = (2/n) * np.sum(e * x**2)

#		Calcul de la descente de gradient : 
#			- On souhaite réduire l'erreur donc nous devons aller dans le sens inverse de la dérivée.
#			- Ici nos dérivées sont positives donc la descente de gradient adaptée est la suivante :
#			- Par rapport à a nous avons  a ← a − η ∂L/∂a autrement dit a = a - epsilon * dL_da
#			- Si la dérivée était négative nous aurions eu a = a + epsilon * dL_da

# Initialisation
n = len(house_prices_df)
a = np.random.randn()
b = np.random.randn()
c = np.random.randn()
x = house_prices_df["surface"]
y = house_prices_df["prix"]
epsilon = 0.0001

# Fonction
def backpropagation_quadratic(a, b, c, x, y, epsilon):
    n = len(x)
    e = a*x**2 + b*x + c - y
    dL_da = (2/n) * np.sum(e * x**2)
    dL_db = (2/n) * np.sum(e * x)
    dL_dc = (2/n) * np.sum(e)
    a = a - epsilon * dL_da
    b = b - epsilon * dL_db
    c = c - epsilon * dL_dc
    modelqdr = quadratic_regression(a, b, c, x)
    rmse_cal = rmse(a, b, c, x, y)
    return a, b, c, modelqdr, rmse_cal

# Boucle d'ebtrainement
# Initialisation
a = np.random.randn()
b = np.random.randn()
c = np.random.randn()

# Fonction
def gradient_descent_quadratic(a, b, c, x, y, epsilon, n_epochs, seuil=0.0000001):
    n = len(x)
    rmse_list = []
    rmse_prev = float('inf')
    for epoch in range(n_epochs):
        a, b, c, modelqdr, rmse_cal = backpropagation_quadratic(a, b, c, x, y, epsilon)
        rmse_list.append(rmse_cal)
        print(f"Epoch {epoch} - RMSE: {rmse_cal:.4f}")
        if abs(rmse_prev - rmse_cal) < seuil:
            break
        rmse_prev = rmse_cal
    return a, b, c, rmse_list

a, b, c, rmse_list = gradient_descent_quadratic(a, b, c, x, y, epsilon, 100000)

x_sorted = x.sort_values()
prediction_sorted = quadratic_regression(a, b, c, x_sorted)
plt.plot(x_sorted, prediction_sorted, color="red", label="régression quadratique")
plt.scatter(x, y, label="données")
plt.legend()
plt.show()

plt.plot(rmse_list, color="blue", label="RMSE")
plt.xlabel("Epochs")
plt.ylabel("RMSE")
plt.legend()
plt.show()






