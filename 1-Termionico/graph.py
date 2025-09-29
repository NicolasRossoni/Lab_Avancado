import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit

# Configurar matplotlib para melhor visualização
plt.rcParams['figure.figsize'] = (10, 8)
plt.rcParams['font.size'] = 16.8

# Ler os dados do CSV
df = pd.read_csv('data.csv', delimiter=',', decimal=',')

# Extrair as temperaturas da primeira linha (exceto a primeira coluna)
temperaturas_celsius = df.columns[1:].tolist()
print(f"Temperaturas em Celsius encontradas: {temperaturas_celsius}")

# Converter temperaturas de Celsius para Kelvin (arredondado para inteiros)
temperaturas = [str(int(round(float(temp) + 273))) for temp in temperaturas_celsius]
print(f"Temperaturas convertidas para Kelvin: {temperaturas}")

# Função para converter strings com vírgula para float
def convert_to_float(value):
    if isinstance(value, str):
        return float(value.replace(',', '.'))
    return float(value)

# Função exponencial com assíntota: y = A * (1 - exp(-B*x)) + C
def exponential_asymptote(x, A, B, C):
    """
    Função exponencial que cresce e depois estagna em uma assíntota
    A: amplitude da curva
    B: taxa de crescimento exponencial
    C: valor da assíntota (onde a curva estagna)
    """
    return A * (1 - np.exp(-B * x)) + C

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
    
    # Cor para esta temperatura
    color = cores[i % len(cores)]
    
    # Fit exponencial com assíntota
    if len(tensao_valid) > 3:  # Precisamos de pelo menos 4 pontos para o fit
        try:
            # Estimar parâmetros iniciais
            y_max = np.max(corrente_valid)
            y_min = np.min(corrente_valid)
            A_init = y_max - y_min  # amplitude estimada
            B_init = 1.0  # taxa de crescimento inicial
            C_init = y_min  # assíntota inicial
            
            # Realizar o fit
            popt, pcov = curve_fit(exponential_asymptote, tensao_valid, corrente_valid, 
                                 p0=[A_init, B_init, C_init],
                                 maxfev=5000,
                                 bounds=([0, 0, -np.inf], [np.inf, np.inf, np.inf]))
            
            A_fit, B_fit, C_fit = popt
            
            # Calcular corrente de saturação (onde a curva converge)
            I_sat = C_fit + A_fit
            
            # Plot dos pontos com corrente de saturação na legenda
            ax.scatter(tensao_valid, corrente_valid, 
                      color=color, s=30, alpha=0.8, 
                      label=f'T = {temp} K ($I_{{sat}}$ = {I_sat:.3f} mA)')
            
            # Gerar pontos para a curva de fit
            x_fit = np.linspace(tensao_valid.min(), tensao_valid.max(), 200)
            y_fit = exponential_asymptote(x_fit, A_fit, B_fit, C_fit)
            
            # Plot da curva de fit
            ax.plot(x_fit, y_fit, color=color, linestyle='-', linewidth=2, alpha=0.7)
            
            # Calcular R² para avaliar a qualidade do fit
            y_pred = exponential_asymptote(tensao_valid, A_fit, B_fit, C_fit)
            ss_res = np.sum((corrente_valid - y_pred) ** 2)
            ss_tot = np.sum((corrente_valid - np.mean(corrente_valid)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)
            
            print(f"Temperatura {temp} K:")
            print(f"  A = {A_fit:.4f}, B = {B_fit:.4f}, C = {C_fit:.4f}")
            print(f"  R² = {r_squared:.4f}")
            print(f"  Corrente de saturação: {I_sat:.4f} mA")
            
        except Exception as e:
            print(f"Erro no fit para temperatura {temp} K: {e}")
            # Fallback: plot sem valor de saturação se o fit falhar
            ax.scatter(tensao_valid, corrente_valid, 
                      color=color, s=30, alpha=0.8, 
                      label=f'T = {temp} K (fit falhou)')
            
            # Fallback para fit linear se o exponencial falhar
            slope, intercept, r_value, p_value, std_err = stats.linregress(tensao_valid, corrente_valid)
            x_fit = np.linspace(tensao_valid.min(), tensao_valid.max(), 100)
            y_fit = slope * x_fit + intercept
            ax.plot(x_fit, y_fit, color=color, linestyle='--', linewidth=2, alpha=0.7)
    else:
        # Plot sem fit se não há pontos suficientes
        ax.scatter(tensao_valid, corrente_valid, 
                  color=color, s=30, alpha=0.8, 
                  label=f'T = {temp:.0f} K (poucos pontos)')

# Configurar o gráfico
ax.set_xlabel('Tensão (V)', fontsize=32.032)
ax.set_ylabel('Corrente (mA)', fontsize=32.032)
ax.set_title('Corrente vs Tensão para diferentes temperaturas', fontsize=32.032, pad=20)

# Configurar a legenda
ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=True, fontsize=19.76)

# Configurar grade
ax.grid(True, alpha=0.3)

# Configurar limites dos eixos para melhor visualização
ax.set_xlim(0, max(tensao) * 1.05)
ax.set_ylim(0, max(corrente) * 1.4)

# Melhorar o layout
plt.tight_layout()

# Salvar o gráfico
plt.savefig('data1.png', dpi=300, bbox_inches='tight')

print("Gráfico gerado com sucesso!")
print("Arquivo salvo como: data1.png")