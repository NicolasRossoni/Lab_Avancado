import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit

# Configurar matplotlib para melhor visualização
plt.rcParams['figure.figsize'] = (10, 8)
plt.rcParams['font.size'] = 16.8

# Função para converter strings com vírgula para float
def convert_to_float(value):
    if isinstance(value, str):
        return float(value.replace(',', '.'))
    return float(value)

# Ler os dados do CSV
df = pd.read_csv('data3.csv', delimiter=',')

# Extrair os nomes das colunas para os rótulos dos eixos
x_label_original = df.columns[0]  # Temperatura (°C)
y_label = df.columns[1]  # Corrente (mA)

# Atualizar rótulo do eixo X para Kelvin
x_label = "Temperatura (K)"  # Modificado para Kelvin

print(f"Eixo X original: {x_label_original}")
print(f"Eixo X modificado: {x_label}")
print(f"Eixo Y: {y_label}")

# Extrair dados e converter para float
x_data_celsius = df.iloc[:, 0].apply(convert_to_float).values  # Temperatura (°C) - eixo x
y_data = df.iloc[:, 1].apply(convert_to_float).values  # Corrente (mA) - eixo y

# Converter temperatura de Celsius para Kelvin
x_data = x_data_celsius + 273  # Converter para Kelvin

print(f"Dados X em Celsius (primeiros 5): {x_data_celsius[:5]}")
print(f"Dados X em Kelvin (primeiros 5): {x_data[:5]}")
print(f"Dados Y (primeiros 5): {y_data[:5]}")

# Remover valores NaN se houver
valid_idx = ~(np.isnan(x_data) | np.isnan(y_data))
x_valid = x_data[valid_idx]
y_valid = y_data[valid_idx]

# Criar o gráfico
fig, ax = plt.subplots(1, 1, figsize=(12, 8))

# Plot dos pontos (vermelho para seguir identidade visual)
ax.scatter(x_valid, y_valid, 
          color='red', s=50, alpha=0.8, 
          label='Dados', marker='o')

# Definir função exponencial para o ajuste
def exponential_func(x, a, b, c):
    return a * np.exp(b * x) + c

# Ajuste exponencial
if len(x_valid) > 3:
    try:
        # Calcular ajuste exponencial
        popt, pcov = curve_fit(exponential_func, x_valid, y_valid, 
                              p0=[0.001, 0.001, 0], maxfev=5000)
        
        a, b, c = popt
        print(f"Parâmetros do ajuste exponencial:")
        print(f"a = {a:.6f}")
        print(f"b = {b:.6f}")
        print(f"c = {c:.6f}")
        
        # Calcular R² para o ajuste exponencial
        y_pred = exponential_func(x_valid, *popt)
        ss_res = np.sum((y_valid - y_pred) ** 2)
        ss_tot = np.sum((y_valid - np.mean(y_valid)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        
        print(f"R² do ajuste exponencial: {r_squared:.4f}")
        
        # Gerar pontos para a linha de fit
        x_fit = np.linspace(x_valid.min(), x_valid.max(), 100)
        y_fit = exponential_func(x_fit, *popt)
        
        # Plot da linha de fit (amarelo para seguir identidade visual)
        ax.plot(x_fit, y_fit, color='gold', linestyle='-', linewidth=2, 
               label='Ajuste exponencial')
        
        fit_success = True
    except Exception as e:
        print(f"Erro no ajuste exponencial: {e}")
        fit_success = False

# Configurar o gráfico
ax.set_xlabel(x_label, fontsize=19.76)
ax.set_ylabel(y_label, fontsize=19.76)

# Criar título com informações do ajuste exponencial
if len(x_valid) > 3 and fit_success:
    # Formatear a equação exponencial
    eq_str = f"y = {a:.2e}·exp({b:.2e}·x) + {c:.2e}"
    ax.set_title('Verificando Lei de Richardson-Dushman', fontsize=22.4, pad=20)
else:
    ax.set_title('Verificando Lei de Richardson-Dushman', fontsize=22.4, pad=20)

# Configurar a legenda
ax.legend(loc='best', frameon=True, fancybox=True, shadow=True, fontsize=19.76)

# Configurar grade
ax.grid(True, alpha=0.3)

# Melhorar o layout
plt.tight_layout()

# Salvar o gráfico
plt.savefig('data3.png', dpi=300, bbox_inches='tight')

print("Gráfico gerado com sucesso!")
print("Arquivo salvo como: data3.png")
