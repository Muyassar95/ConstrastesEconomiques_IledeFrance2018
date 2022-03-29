
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import preprocessing
import statsmodels.api as sm

data_communes = pd.read_csv("C:/Users/almou/OneDrive/Documents/Inégalités France/Inegalites2018_IDFCommunes.csv", sep=';')
data_communes

data_communes.describe()

communes_socio = data_communes.copy()

data_communes.columns

X = data_communes[[
       'Part Pop 15 ans ou plus non scol. Sans diplôme ou CEP ',
       'Part Pop 15 ans ou plus non scol. BEPC, brevet des collèges, DNB ',
       'Part Pop 15 ans ou plus non scol. CAP-BEP ou équiv. ',
       'Part Pop 15 ans ou plus non scol. Bac, brevet pro. ou équiv. ',
       'Part Pop 15 ans ou plus Enseignement sup', "Taux d'emploi 15-64 ans",
       'Taux de chômage 15-64 ans',
       'Part Actifs 15-64 ans Agriculteurs exploitants  ',
       'Part Actifs 15-64 ans Artisans, Comm., Chefs entr.  ',
       'Part Actifs 15-64 ans Cadres, Prof. intel. sup.  ',
       'Part Actifs 15-64 ans Prof. intermédiaires  ',
       'Part Actifs 15-64 ans Employés  ', 'Part Actifs 15-64 ans Ouvriers  ',
       "Taux d'immigration"]]

Y = data_communes[['Médiane (€)']]

corr_communesocio = X.corr(method='pearson')

plt.figure(figsize=(20,20))
sns.heatmap(corr_communesocio, annot=True)
plt.show()

reg = LinearRegression()
reg.fit(X,Y)

variables_socioeco = pd.DataFrame({'Variables explicatives':X.columns,'Coefficients':reg.coef_[0]})
variables_socioeco.append({'Variables explicatives':'Constante','Coefficients':reg.intercept_[0]},ignore_index=True).set_index('Variables explicatives')

y_pred = pd.DataFrame(reg.predict(X),dtype='float').set_axis(['Prédictions (Revenu médian)'],axis=1)

communes = pd.concat([data_communes,y_pred],axis=1)
communes

print('Erreur-type du revenu médian = ',np.sqrt(mean_squared_error(Y,y_pred)))

print('Coefficient de détermination R2 = ', round(r2_score(Y,y_pred),3))

plt.scatter(Y,y_pred)
plt.title('Les valeurs prédites contre les valeurs observées')
plt.plot(y_pred,y_pred,c='red',label='Valeurs prédites')
plt.xlabel('Valeurs observées')
plt.ylabel('Valeurs prédites')
plt.legend()
plt.show()
