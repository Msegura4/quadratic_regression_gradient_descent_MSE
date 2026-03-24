import pandas
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    house_prices_df = pandas.read_csv("prix_maisons.csv")
    x_mean, x_std = house_prices_df["surface"].mean(), house_prices_df["surface"].std()
    y_mean, y_std = house_prices_df["prix"].mean(), house_prices_df["prix"].std()
    house_prices_df["surface"] = (house_prices_df["surface"] - x_mean) / x_std
    house_prices_df["prix"] = (house_prices_df["prix"] - y_mean) / y_std

    n = len(house_prices_df)
    a = np.random.randn()
    b = np.random.randn()
    epsilon = 0.0001
    mse_prev = float('inf')
    seuil = 0.0000001

    while True:
        a = a - epsilon * (2/n) * np.sum((a * house_prices_df["surface"] + 
            b - house_prices_df["prix"]) * house_prices_df["surface"])
        b = b - epsilon * (2/n) * np.sum((a * house_prices_df["surface"] + b - house_prices_df["prix"]))
        mse = (1/n) * np.sum((a * house_prices_df["surface"] + b - house_prices_df["prix"])**2)
        print(f"MSE: {mse:.4f} - a: {a:.4f} - b: {b:.4f}")
        if abs(mse_prev - mse) < seuil:
            break
        mse_prev = mse

    plt.plot(house_prices_df["surface"], a * house_prices_df["surface"] + b, color="green", label="régression linéaire")
    plt.scatter(house_prices_df["surface"], house_prices_df["prix"], label="données")
    plt.legend()
    plt.show()