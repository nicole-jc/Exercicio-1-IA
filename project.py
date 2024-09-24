import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

df = pd.read_csv('datapy.csv')
colunasX = ["Categoria", "Passageiros", "PortaMalas", "ArCondicionado", "Cambio"]
colunasY = ["Valor"]
dadosX = df[colunasX]
dadosY = df[colunasY]
modelo = LinearRegression().fit(dadosX, dadosY)

def solicitar_dados():
    print("Selecione a categoria do carro:")
    print("1: Econômico")
    print("2: Intermediário")
    print("3: Compacto")
    print("4: SUV")
    print("5: Premium")
    categoria = int(input("Categoria: "))

    passageiros = int(input("Número de passageiros (5): "))
    porta_malas = int(input("Capacidade do porta-malas (1, 2 ou 3): "))
    ar_condicionado = int(input("Ar condicionado (1: Não, 2: Sim): "))
    cambio = int(input("Câmbio (1: Manual, 2: Automático): "))

    return [categoria, passageiros, porta_malas, ar_condicionado, cambio]

def main():
    dados_usuario = solicitar_dados()
    valoresTeste = np.array([dados_usuario])
    predicao = modelo.predict(valoresTeste)

    print(f"Predição do Valor: {predicao[0][0]:.2f}")

if __name__ == "__main__":
    main()
