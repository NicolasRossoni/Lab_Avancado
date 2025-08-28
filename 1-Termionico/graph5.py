import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats

# Configurar matplotlib para melhor visualização
plt.rcParams['figure.figsize'] = (10, 8)
plt.rcParams['font.size'] = 12

# Função para converter strings com vírgula para float
def convert_to_float(value):
    if isinstance(value, str):
        return float(value.replace(',', '.'))
    return float(value)

# Função que modela o comportamento: assíntota no início, depois polinômio
def piecewise_asymptotic_function(x, x_corte, y_assintota, a, *poly_coeffs):
    """
    Função por partes:
    - Exponencial assintótica para x < x_corte: y_assintota * (1 - exp(-a*x))
    - Polinômio para x >= x_corte
    """
    # Parte exponencial assintótica
    parte1 = y_assintota * (1 - np.exp(-a * x))
    # Parte polinomial (aplicada aos pontos após x_corte)
    x_shifted = x - x_corte
    parte2 = np.polyval(poly_coeffs, x_shifted)
    
    return np.where(x < x_corte, parte1, parte2)

def detect_cutoff_point(x_data, y_data, threshold=0.9):
    """
    Detecta automaticamente o ponto de corte onde começa a queda
    """
    if len(x_data) < 5:
        return len(x_data) // 3
    
    # Procurar pelo primeiro ponto onde y começa a diminuir significativamente
    y_max = np.max(y_data)
    for i in range(len(y_data)):
        if y_data[i] < threshold * y_max:
            return max(0, i - 1)  # Pegar um ponto antes da queda
    
    return len(x_data) // 2  # Fallback para metade dos dados

def calcular_interseccao(y_horizontal, slope, intercept):
    """
    Calcula a interseçção entre uma reta horizontal e uma reta inclinada
    y_horizontal = slope * x + intercept
    x = (y_horizontal - intercept) / slope
    """
    if abs(slope) < 1e-10:  # Evitar divisão por zero
        return None
    return (y_horizontal - intercept) / slope

# Lista dos arquivos CSV e suas respectivas temperaturas
dados = [
    {'arquivo': 'data5_1.csv', 'temp_celsius': 1907, 'nome': 'data5_1'},
    {'arquivo': 'data5_2.csv', 'temp_celsius': 1765, 'nome': 'data5_2'},
    {'arquivo': 'data5_3.csv', 'temp_celsius': 1630, 'nome': 'data5_3'}
]

# Processar cada arquivo CSV e gerar gráfico individual
for i, info in enumerate(dados):
    arquivo = info['arquivo']
    temp_celsius = info['temp_celsius']
    nome = info['nome']
    
    # Converter temperatura para Kelvin
    temp_kelvin = temp_celsius + 273
    
    print(f"Processando {arquivo}...")
    print(f"  Temperatura: {temp_celsius}°C = {temp_kelvin} K")
    
    # Ler dados do CSV
    data = pd.read_csv(arquivo, delimiter=',', header=0)
    
    # Extrair colunas (assumindo que são as duas primeiras)
    x_data = data.iloc[:, 0].astype(str).apply(convert_to_float) * 1000  # Converter Tesla para miliTesla
    y_data = data.iloc[:, 1].astype(str).apply(convert_to_float)  # Já em micro Ampere
    
    print(f"  Dados X (primeiros 5): {x_data[:5]}")
    print(f"  Dados Y (primeiros 5): {y_data[:5]}")
    
    # Remover valores NaN se houver
    valid_idx = ~(np.isnan(x_data) | np.isnan(y_data))
    x_valid = x_data[valid_idx]
    y_valid = y_data[valid_idx]
    
    # Criar gráfico individual
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Primeira reta azul: média dos 5 primeiros pontos (assíntota inicial)
    y_assintota_inicial = np.mean(y_valid[:5]) if len(y_valid) >= 5 else np.mean(y_valid)
    
    # Plot dos pontos (vermelho, sem conectar)
    ax.scatter(x_valid, y_valid, 
              color='red', s=50, alpha=0.8, 
              label='Dados', marker='o')
    
    # Fit polinomial com pontos fake nos primeiros 20% (amarela)
    try:
        # Encontrar índice correspondente aos primeiros 20% do eixo x
        x_range = x_valid.max() - x_valid.min()
        x_20_percent = x_valid.min() + 0.20 * x_range
        
        # Calcular média dos pontos nos primeiros 20%
        mask_inicio = x_valid <= x_20_percent
        if np.any(mask_inicio):
            y_media_inicio = np.mean(y_valid[mask_inicio])
            print(f"  Média dos primeiros 20%: {y_media_inicio:.3f}")
            
            # Criar pontos fake no intervalo dos primeiros 20%
            n_pontos_fake = 20  # Número de pontos fake a adicionar
            x_fake = np.linspace(x_valid.min(), x_20_percent, n_pontos_fake)
            y_fake = np.full(n_pontos_fake, y_media_inicio)  # Todos com valor da média
            
            # Combinar dados reais com pontos fake
            x_combined = np.concatenate([x_fake, x_valid])
            y_combined = np.concatenate([y_fake, y_valid])
            
            print(f"  Adicionados {n_pontos_fake} pontos fake com valor {y_media_inicio:.3f}")
        else:
            # Se não houver pontos nos primeiros 20%, usar dados originais
            x_combined = x_valid
            y_combined = y_valid
        
        # Realizar fit polinomial de grau 3 com dados combinados
        poly_coeffs = np.polyfit(x_combined, y_combined, 3)
        
        # Gerar pontos para a curva de fit (amarela)
        x_fit = np.linspace(x_valid.min(), x_valid.max(), 200)
        y_fit = np.polyval(poly_coeffs, x_fit)
        
        # Plot da curva de fit (amarela)
        ax.plot(x_fit, y_fit, color='gold', linestyle='-', linewidth=2, 
               label='Fit polinomial', alpha=0.8)
        
        print(f"  Coeficientes do polinômio: {poly_coeffs}")
        
    except Exception as e:
        print(f"  Erro no fit polinomial: {e}")
    
    # Primeira reta azul: assíntota horizontal inicial (linha cheia, mais curta) - SEM LEGENDA
    # Tornar a reta mais curta (até 70% do range)
    x_end_assintota = x_valid.min() + 0.7 * (x_valid.max() - x_valid.min())
    x_assintota_inicial = np.linspace(x_valid.min(), x_end_assintota, 100)
    y_assintota_inicial_line = np.full_like(x_assintota_inicial, y_assintota_inicial)
    ax.plot(x_assintota_inicial, y_assintota_inicial_line, color='blue', linestyle='-', 
           linewidth=2, alpha=0.7)
    
    # Segunda reta azul: fit linear dos pontos após 40% do eixo x (10 pontos)
    slope_final = None
    intercept_final = None
    try:
        # Encontrar índice correspondente a 40% do range de x
        x_range = x_valid.max() - x_valid.min()
        x_40_percent = x_valid.min() + 0.4 * x_range
        
        # Encontrar pontos após 40% do eixo x
        indices_depois = x_valid >= x_40_percent
        x_depois = x_valid[indices_depois]
        y_depois = y_valid[indices_depois]
        
        # Pegar até 10 pontos (ou todos se forem menos que 10)
        if len(x_depois) > 0:
            n_pontos = min(10, len(x_depois))
            x_fit_final = x_depois[:n_pontos]
            y_fit_final = y_depois[:n_pontos]
            
            if len(x_fit_final) >= 2:  # Precisamos de pelo menos 2 pontos para fit linear
                # Fit linear
                slope_final, intercept_final = np.polyfit(x_fit_final, y_fit_final, 1)
                
                # Calcular interseção com a primeira reta azul
                x_interseccao = calcular_interseccao(y_assintota_inicial, slope_final, intercept_final)
                
                # Estender a reta para a esquerda até passar um pouco da altura da primeira reta
                if x_interseccao is not None and x_interseccao > x_valid.min():
                    # Começar um pouco antes da interseçção
                    x_inicio = x_interseccao - 0.1 * x_range
                else:
                    x_inicio = x_valid.min()
                
                # Gerar pontos para a segunda reta azul (estendida)
                x_reta_final = np.linspace(x_inicio, x_valid.max(), 100)
                y_reta_final = slope_final * x_reta_final + intercept_final
                
                # Plot da segunda reta azul - SEM LEGENDA
                ax.plot(x_reta_final, y_reta_final, color='blue', linestyle='-', 
                       linewidth=2, alpha=0.7)
                
                # Reta vertical tracejada e ponto preto na interseçção (Bcorte)
                if x_interseccao is not None:
                    # Encontrar o ponto vermelho mais baixo
                    y_min_vermelho = np.min(y_valid)
                    
                    # Reta vertical tracejada do ponto vermelho mais baixo até o ponto de interseçção
                    ax.plot([x_interseccao, x_interseccao], [y_min_vermelho, y_assintota_inicial], 
                           color='black', linestyle='--', linewidth=2, alpha=0.8)
                    
                    # Ponto preto na interseçção
                    ax.plot(x_interseccao, y_assintota_inicial, 'ko', markersize=8)
                    
                    # Salvar valor de Bcorte para usar no texto
                    bcorte_value = x_interseccao
                    print(f"  Bcorte (interseçção das retas): {x_interseccao:.4f}")
                    print(f"  Reta preta vai de y={y_min_vermelho:.3f} até y={y_assintota_inicial:.3f}")
                
                print(f"  Segunda reta azul - slope: {slope_final:.3f}, intercept: {intercept_final:.3f}")
        
    except Exception as e:
        print(f"  Erro na segunda reta azul: {e}")
    
    # Configurar o gráfico com novos rótulos
    ax.set_xlabel('Campo Magnético (mT)', fontsize=14)
    ax.set_ylabel('Corrente entre elétrons (μA)', fontsize=14)
    ax.set_title(f'Determinação do Campo Magnético de corte $\\mathbf{{B}}_{{corte}}$\nT = {temp_kelvin} K', 
                fontsize=16, pad=20)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=12)
    
    # Adicionar texto com Bcorte e e/m abaixo da legenda
    try:
        if 'bcorte_value' in locals():
            # Cálculo da relação e/m
            # e/m = (2*4.81)/((B_corte*0.02)^2)/(10^(11))
            em_ratio = (2 * 4.81) / ((bcorte_value * 0.02)**2) / (10**11)
            
            # Posicionar texto com Bcorte (maior, B em negrito)
            ax.text(0.98, 0.65, f'$\\mathbf{{B}}_{{corte}}$ = {bcorte_value:.3f} mT', 
                   transform=ax.transAxes, fontsize=16, 
                   horizontalalignment='right', verticalalignment='top',
                   style='italic', color='black')
            
            # Posicionar texto com e/m embaixo do Bcorte (maior)
            ax.text(0.98, 0.60, f'e/m = {em_ratio:.2e} C/kg', 
                   transform=ax.transAxes, fontsize=14, 
                   horizontalalignment='right', verticalalignment='top',
                   style='italic', color='black')
    except:
        pass
    
    # Salvar o gráfico individual
    nome_arquivo = f'{nome}.png'
    plt.savefig(nome_arquivo, dpi=300, bbox_inches='tight')
    plt.close()  # Fechar figura para liberar memória
    
    print(f"  Gráfico salvo como: {nome_arquivo}")
    print()

print("Todos os gráficos individuais foram gerados com sucesso!")
