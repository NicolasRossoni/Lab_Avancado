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

# Ajuste linear
if len(x_valid) > 1:
    # Calcular regressão linear
    slope, intercept, r_value, p_value, std_err = stats.linregress(x_valid, y_valid)
    
    print(f"Coeficiente angular: {slope:.4f}")
    print(f"Intercepto: {intercept:.4f}")
    print(f"R²: {r_value**2:.4f}")
    print(f"Erro padrão do slope: {std_err:.4f}")
    
    # Gerar pontos para a linha de fit
    x_fit = np.linspace(x_valid.min(), x_valid.max(), 100)
    y_fit = slope * x_fit + intercept
    
    # Plot dos pontos (em vermelho)
    ax.scatter(x_valid, y_valid, 
              color='red', s=50, alpha=0.8, 
              label='Dados', marker='o')
    
    # Plot da linha de fit (em amarelo)
    ax.plot(x_fit, y_fit, color='gold', linestyle='-', linewidth=2, 
           label='Fit linear')

# Configurar o gráfico
ax.set_xlabel(x_label, fontsize=32.032)
ax.set_ylabel(y_label, fontsize=32.032)
ax.set_title('Verificação da lei de Child-Langmuir', fontsize=32.032, pad=20)

# Configurar a legenda no topo esquerdo
ax.legend(loc='center left', frameon=True, fancybox=True, shadow=True, fontsize=19.76)

# Adicionar equação da reta em itálico abaixo da legenda
if len(x_valid) > 1:
    # Primeira linha: equação
    equation_text = f'y = {slope:.4f}x - {-intercept:.4f}'
    ax.text(0.02, 0.85, equation_text, transform=ax.transAxes, 
            fontsize=22.4, style='italic', color='black')
    
    # Segunda linha: coeficiente linear com incerteza
    coef_text = f'Coef. lin. {slope:.4f} ± {std_err:.4f}'
    ax.text(0.02, 0.80, coef_text, transform=ax.transAxes, 
            fontsize=22.4, style='italic', color='black')

# Configurar grade
ax.grid(True, alpha=0.3)

# Melhorar o layout
plt.tight_layout()

# Salvar o gráfico
plt.savefig('data2.png', dpi=300, bbox_inches='tight')

print("Gráfico gerado com sucesso!")
print("Arquivo salvo como: data2.png")
