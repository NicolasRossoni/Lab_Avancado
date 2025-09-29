import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Configurar matplotlib para melhor visualização
plt.rcParams['figure.figsize'] = (10, 8)
plt.rcParams['font.size'] = 16.8

# Função para converter strings com vírgula para float
def convert_to_float(value):
    if isinstance(value, str):
        return float(value.replace(',', '.'))
    return float(value)

# Constante de Boltzmann
Kb = 8.61e-5  # eV/K

# Ler os dados do CSV
df = pd.read_csv('data4.csv', delimiter=',')

# Extrair os nomes das colunas para os rótulos dos eixos
y_label = df.columns[1]  # Log(j/T²)

print(f"Eixo X original: {df.columns[0]}")
print(f"Eixo Y: {y_label}")

# Extrair dados e converter para float
x_data_raw = df.iloc[:, 0].apply(convert_to_float).values  # 1/T em 1/°C
y_data = df.iloc[:, 1].apply(convert_to_float).values  # Log(j/T²) - eixo y

# Converter de 1/T_celsius para 1/T_kelvin
# Se x_data_raw = 1/T_celsius, então T_celsius = 1/x_data_raw
# T_kelvin = T_celsius + 273
# x_data = 1/T_kelvin = 1/(T_celsius + 273) = 1/(1/x_data_raw + 273)
T_celsius = 1/x_data_raw  # Converter de 1/T para T em Celsius
T_kelvin = T_celsius + 273  # Converter para Kelvin
x_data = 1/T_kelvin  # Converter de volta para 1/T em Kelvin

print(f"Temperaturas em Celsius (primeiras 5): {T_celsius[:5]}")
print(f"Temperaturas em Kelvin (primeiras 5): {T_kelvin[:5]}")
print(f"Constante de Boltzmann: Kb = {Kb}")

print(f"Dados X (primeiros 5): {x_data[:5]}")
print(f"Dados Y (primeiros 5): {y_data[:5]}")

# Remover valores NaN se houver
valid_idx = ~(np.isnan(x_data) | np.isnan(y_data))
x_valid = x_data[valid_idx]
y_valid = y_data[valid_idx]

# Criar o gráfico
fig, ax = plt.subplots(1, 1, figsize=(12, 8))

# Ajuste linear
if len(x_valid) > 1:
    # Calcular regressão linear
    slope, intercept, r_value, p_value, std_err = stats.linregress(x_valid, y_valid)
    
    print(f"Coeficiente angular: {slope:.4f}")
    print(f"Intercepto: {intercept:.4f}")
    print(f"R²: {r_value**2:.4f}")
    print(f"Erro padrão do slope: {std_err:.4f}")
    
    # Calcular W (função trabalho) e sua incerteza
    W_value = abs(slope) * Kb  # Valor absoluto porque W é positivo
    W_uncertainty = std_err * Kb  # Incerteza de W
    
    print(f"W = {W_value:.4f} ± {W_uncertainty:.4f} eV")
    
    # Gerar pontos para a linha de fit
    x_fit = np.linspace(x_valid.min(), x_valid.max(), 100)
    y_fit = slope * x_fit + intercept
    
    # Plot dos pontos (em vermelho)
    ax.scatter(x_valid, y_valid, 
              color='red', s=50, alpha=0.8, 
              label='Dados', marker='o')
    
    # Plot da linha de fit (em amarelo)
    ax.plot(x_fit, y_fit, color='gold', linestyle='-', linewidth=2, 
           label='Ajuste linear')

# Configurar o gráfico
ax.set_xlabel('1/T (1/K)', fontsize=32.032)
ax.set_ylabel(y_label, fontsize=32.032)
ax.set_title('Descobrindo W do Tungstênio', fontsize=32.032, pad=20)

# Configurar a legenda no topo esquerdo
ax.legend(loc='center right', frameon=True, fancybox=True, shadow=True, fontsize=19.76)

# Adicionar equação da reta e valor de W em itálico
if len(x_valid) > 1:
    equation_text = f'Coef. lin. y = {slope:.4f}x + {intercept:.4f}'
    ax.text(0.98, 0.85, equation_text, transform=ax.transAxes, 
            fontsize=22.4, style='italic', ha='right', color='black')
    
    # Adicionar valor de W com incerteza
    W_text = f'W = ({W_value:.4f} ± {W_uncertainty:.4f}) eV'
    ax.text(0.98, 0.80, W_text, transform=ax.transAxes, 
            fontsize=22.4, style='italic', ha='right', color='black')

# Configurar grade
ax.grid(True, alpha=0.3)

# Melhorar o layout
plt.tight_layout()

# Salvar o gráfico
plt.savefig('data4.png', dpi=300, bbox_inches='tight')

print("Gráfico gerado com sucesso!")
print("Arquivo salvo como: data4.png")
