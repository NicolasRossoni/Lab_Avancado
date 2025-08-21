import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Configurar matplotlib para melhor visualização
plt.rcParams['figure.figsize'] = (10, 8)
plt.rcParams['font.size'] = 12

# Função para converter strings com vírgula para float
def convert_to_float(value):
    if isinstance(value, str):
        return float(value.replace(',', '.'))
    return float(value)

# Ler os dados do CSV
df = pd.read_csv('data2.csv', delimiter=',')

# Extrair os nomes das colunas para os rótulos dos eixos
x_label = df.columns[0]  # Log(V)
y_label = df.columns[1]  # Log(j)

print(f"Eixo X: {x_label}")
print(f"Eixo Y: {y_label}")

# Extrair dados e converter para float
x_data = df.iloc[:, 0].apply(convert_to_float).values  # Log(V) - eixo x
y_data = df.iloc[:, 1].apply(convert_to_float).values  # Log(j) - eixo y

print(f"Dados X (primeiros 5): {x_data[:5]}")
print(f"Dados Y (primeiros 5): {y_data[:5]}")

# Remover valores NaN se houver
valid_idx = ~(np.isnan(x_data) | np.isnan(y_data))
x_valid = x_data[valid_idx]
y_valid = y_data[valid_idx]

# Criar o gráfico
fig, ax = plt.subplots(1, 1, figsize=(12, 8))

# Plot dos pontos
ax.scatter(x_valid, y_valid, 
          color='blue', s=50, alpha=0.8, 
          label='Dados experimentais', marker='o')

# Ajuste linear
if len(x_valid) > 1:
    # Calcular regressão linear
    slope, intercept, r_value, p_value, std_err = stats.linregress(x_valid, y_valid)
    
    print(f"Coeficiente angular: {slope:.4f}")
    print(f"Intercepto: {intercept:.4f}")
    print(f"R²: {r_value**2:.4f}")
    
    # Gerar pontos para a linha de fit
    x_fit = np.linspace(x_valid.min(), x_valid.max(), 100)
    y_fit = slope * x_fit + intercept
    
    # Plot da linha de fit
    ax.plot(x_fit, y_fit, color='red', linestyle='-', linewidth=2, 
           label=f'Ajuste linear (m = {slope:.4f})')

# Configurar o gráfico
ax.set_xlabel(x_label, fontsize=14)
ax.set_ylabel(y_label, fontsize=14)
ax.set_title(f'Gráfico Log-Log - Coeficiente Angular = {slope:.4f}', fontsize=16, pad=20)

# Configurar a legenda
ax.legend(loc='best', frameon=True, fancybox=True, shadow=True)

# Configurar grade
ax.grid(True, alpha=0.3)

# Melhorar o layout
plt.tight_layout()

# Salvar o gráfico
plt.savefig('data2.png', dpi=300, bbox_inches='tight')

print("Gráfico gerado com sucesso!")
print("Arquivo salvo como: data2.png")
