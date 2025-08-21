import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Configurar matplotlib para melhor visualização
plt.rcParams['figure.figsize'] = (10, 8)
plt.rcParams['font.size'] = 12

# Ler os dados do CSV
df = pd.read_csv('data.csv', delimiter=',', decimal=',')

# Extrair as temperaturas da primeira linha (exceto a primeira coluna)
temperaturas = df.columns[1:].tolist()
print(f"Temperaturas encontradas: {temperaturas}")

# Função para converter strings com vírgula para float
def convert_to_float(value):
    if isinstance(value, str):
        return float(value.replace(',', '.'))
    return float(value)

# Extrair tensão (primeira coluna de dados)
tensao = df.iloc[:, 0].apply(convert_to_float).values

# Criar o gráfico
fig, ax = plt.subplots(1, 1, figsize=(12, 8))

# Cores para cada temperatura (similar à imagem)
cores = ['green', 'darkgreen', 'olive', 'orange', 'orangered', 'red', 'darkred']

# Plot para cada temperatura
for i, temp in enumerate(temperaturas):
    # Extrair corrente para esta temperatura
    corrente = df.iloc[:, i+1].values
    
    # Converter para float caso necessário
    corrente = df.iloc[:, i+1].apply(convert_to_float).values
    
    # Remover valores NaN
    valid_idx = ~np.isnan(corrente)
    tensao_valid = tensao[valid_idx]
    corrente_valid = corrente[valid_idx]
    
    # Plot dos pontos
    color = cores[i % len(cores)]
    ax.scatter(tensao_valid, corrente_valid, 
              color=color, s=30, alpha=0.8, 
              label=f'T = {temp} K')
    
    # Linha de fit (regressão linear)
    if len(tensao_valid) > 1:
        # Calcular regressão linear
        slope, intercept, r_value, p_value, std_err = stats.linregress(tensao_valid, corrente_valid)
        
        # Gerar pontos para a linha de fit
        x_fit = np.linspace(tensao_valid.min(), tensao_valid.max(), 100)
        y_fit = slope * x_fit + intercept
        
        # Plot da linha de fit
        ax.plot(x_fit, y_fit, color=color, linestyle='-', linewidth=2, alpha=0.7)

# Configurar o gráfico
ax.set_xlabel('Tensão (V)', fontsize=14)
ax.set_ylabel('Corrente (mA)', fontsize=14)
ax.set_title('Corrente vs Tensão para diferentes temperaturas', fontsize=16, pad=20)

# Configurar a legenda
ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)

# Configurar grade
ax.grid(True, alpha=0.3)

# Configurar limites dos eixos para melhor visualização
ax.set_xlim(0, max(tensao) * 1.05)
ax.set_ylim(0, None)

# Melhorar o layout
plt.tight_layout()

# Salvar o gráfico
plt.savefig('data1.png', dpi=300, bbox_inches='tight')

# Mostrar o gráfico
plt.show()

print("Gráfico gerado com sucesso!")
print("Arquivo salvo como: data1.png")