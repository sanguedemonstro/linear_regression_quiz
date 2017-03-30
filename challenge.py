import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt

#lendo os dados
bmi_life_data = pd.read_csv('bmi_and_life_expectancy.csv')
x_values = bmi_life_data[['BMI']]
y_values = bmi_life_data[['Life expectancy']]

#treinando o modelo com os dados lidos
bmi_life_model = linear_model.LinearRegression()
bmi_life_model.fit(x_values, y_values)

#gerando um gráfico com os dados e plotando a regressão linear
plt.scatter(x_values, y_values)
plt.plot(x_values, bmi_life_model.predict(x_values))
plt.show()

#prevendo a expectativa de vida para um IMC = 21.07931
laos_life_exp = bmi_life_model.predict([[21.07931]])

#imprimindo a expectativa de vida prevista
print(laos_life_exp[0][0])
