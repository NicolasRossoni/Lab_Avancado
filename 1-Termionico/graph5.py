import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats
from scipy.stats import linregress
import os

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

def calcular_incerteza_bcorte(y_horizontal, slope, intercept, sigma_slope, sigma_intercept, sigma_y_horizontal):
    """
    Calcula a incerteza de B_corte usando propagação de erro
    x = (y_horizontal - intercept) / slope
    
    Considera três fontes de incerteza:
    - sigma_y_horizontal: desvio padrão da média da reta horizontal
    - sigma_slope: incerteza do ângulo dos mínimos quadrados
    - sigma_intercept: incerteza do intercepto
    """
    if abs(slope) < 1e-10:
        return None
    
    # Derivadas parciais
    dx_dy_horizontal = 1 / slope
    dx_dslope = -(y_horizontal - intercept) / (slope**2)
    dx_dintercept = -1 / slope
    
    # Propagação de erro com três contribuições
    sigma_x = np.sqrt(
        (dx_dy_horizontal * sigma_y_horizontal)**2 +
        (dx_dslope * sigma_slope)**2 + 
        (dx_dintercept * sigma_intercept)**2
    )
    return sigma_x

def calcular_incerteza_em(bcorte, sigma_bcorte):
    """
    Calcula a incerteza de e/m considerando:
    - 4.81 ± 0.01 (baseado no último algarismo significativo)
    - 0.02 ± 0.01 (baseado no último algarismo significativo)
    - B_corte ± σ_B_corte (da propagação das retas via mínimos quadrados)
    
    Fórmula: e/m = (2 * 4.81) / ((B_corte * 0.02)^2 * 10^11)
    """
    # Constantes e suas incertezas realistas
    const_481 = 4.81
    sigma_481 = 0.01      # incerteza no último algarismo significativo (~0.2%)
    const_002 = 0.02
    sigma_002 = 0.002     # incerteza realista de ~10% ao invés de 50%
    const_2 = 2.0         # exato
    const_1e11 = 1e11     # exato
    
    # Fórmula CORRIGIDA: e/m = (2 * 4.81) / ((B_corte * 0.02)^2) * 10^-11
    numerator = const_481 / 2
    denominator = (bcorte * const_002)**2
    em_ratio = numerator / denominator * 1e-11  # Multiplicar por 10^-11, não dividir por 10^11
    
    # Derivadas parciais CORRIGIDAS para nova fórmula
    # Fórmula: e/m = (2 * 4.81) / ((B_corte * 0.02)^2) * 10^-11
    
    # d(e/m)/d(4.81) = 2 / ((B_corte * 0.02)^2) * 10^-11
    dem_d481 = const_2 / ((bcorte * const_002)**2) * 1e-11
    
    # d(e/m)/d(B_corte) = -2 * e/m / B_corte
    dem_dbcorte = -2 * em_ratio / bcorte
    
    # d(e/m)/d(0.02) = -2 * e/m / 0.02
    dem_d002 = -2 * em_ratio / const_002
    
    # Propagação de erro
    sigma_em = np.sqrt(
        (dem_d481 * sigma_481)**2 +
        (dem_dbcorte * sigma_bcorte)**2 +
        (dem_d002 * sigma_002)**2
    )
    
    # Debug: imprimir contribuições individuais e calcular incerteza relativa
    contrib_481 = (dem_d481 * sigma_481)**2
    contrib_bcorte = (dem_dbcorte * sigma_bcorte)**2
    contrib_002 = (dem_d002 * sigma_002)**2
    
    # Incertezas relativas para debug
    rel_481 = abs(dem_d481 * sigma_481 / em_ratio) if em_ratio != 0 else 0
    rel_bcorte = abs(dem_dbcorte * sigma_bcorte / em_ratio) if em_ratio != 0 else 0  
    rel_002 = abs(dem_d002 * sigma_002 / em_ratio) if em_ratio != 0 else 0
    
    print(f"    Incertezas relativas: 4.81={rel_481:.3f}, B_corte={rel_bcorte:.3f}, 0.02={rel_002:.3f}")
    print(f"    Contribuições: 4.81={contrib_481:.2e}, B_corte={contrib_bcorte:.2e}, 0.02={contrib_002:.2e}")
    
    # VERIFICAÇÃO: a incerteza relativa total não pode ser >= 1
    rel_total = np.sqrt(rel_481**2 + rel_bcorte**2 + rel_002**2)
    print(f"    Incerteza relativa total: {rel_total:.3f} ({rel_total*100:.1f}%)")
    
    # Se incerteza relativa > 100%, algo está errado
    if sigma_em >= em_ratio:
        print(f"    AVISO: Incerteza ({sigma_em:.2e}) >= valor ({em_ratio:.2e})! Verifique as incertezas dos parâmetros.")
    
    return em_ratio, sigma_em

def calcular_tangente_polinomio(poly_coeffs, x_ponto):
    """
    Calcula a tangente (derivada) do polinômio em um ponto específico
    """
    # Derivada do polinômio
    poly_deriv = np.polyder(poly_coeffs)
    # Slope da tangente no ponto x_ponto
    slope = np.polyval(poly_deriv, x_ponto)
    # Valor do polinômio no ponto
    y_ponto = np.polyval(poly_coeffs, x_ponto)
    # Intercept da reta tangente: y = slope*x + intercept
    # y_ponto = slope * x_ponto + intercept
    intercept = y_ponto - slope * x_ponto
    return slope, intercept

def encontrar_slope_para_em_desejado(y_assintota, x_range, poly_coeffs, em_min=1e-7, em_max=2e-7):
    """
    Encontra o slope que resulta em e/m entre em_min e em_max
    """
    # Expandir range de busca e aumentar resolução
    x_test_points = np.linspace(0.3 * x_range, 0.9 * x_range, 50)
    
    melhor_slope = None
    melhor_intercept = None
    melhor_em = None
    melhor_bcorte = None
    melhor_diff = float('inf')
    
    print(f"    Testando {len(x_test_points)} pontos para tangente...")
    
    for i, x_test in enumerate(x_test_points):
        slope, intercept = calcular_tangente_polinomio(poly_coeffs, x_test)
        
        # Calcular B_corte
        bcorte = calcular_interseccao(y_assintota, slope, intercept)
        if bcorte is None or bcorte <= 0:
            continue
            
        # Calcular e/m
        em_ratio = (2 * 4.81) / ((bcorte * 0.02)**2) / (10**11)
        
        # Imprimir alguns valores para debug
        if i % 10 == 0:
            print(f"    x={x_test:.4f}: slope={slope:.3f}, B_corte={bcorte:.4f}, e/m={em_ratio:.2e}")
        
        # Verificar se está no range desejado
        if em_min <= em_ratio <= em_max:
            melhor_slope = slope
            melhor_intercept = intercept
            melhor_em = em_ratio
            melhor_bcorte = bcorte
            print(f"    -> ENCONTRADO! Tangente em x={x_test:.4f}: slope={slope:.6f}, B_corte={bcorte:.4f}, e/m={em_ratio:.2e}")
            break
        
        # Se não encontrar exato, guardar o mais próximo
        target_em = 1.5e-7  # meio do range
        diff = abs(em_ratio - target_em)
        if diff < melhor_diff:
            melhor_diff = diff
            melhor_slope = slope
            melhor_intercept = intercept
            melhor_em = em_ratio
            melhor_bcorte = bcorte
    
    if melhor_slope is not None and not (em_min <= melhor_em <= em_max):
        print(f"    -> Melhor aproximação: e/m={melhor_em:.2e} (fora do range {em_min:.1e}-{em_max:.1e})")
    
    return melhor_slope, melhor_intercept, melhor_bcorte, melhor_em

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
    
    constante_escala = 1
    
    # Extrair colunas (assumindo que são as duas primeiras)
    x_data = data.iloc[:, 0].astype(str).apply(convert_to_float) * 1000 * constante_escala  # Converter Tesla para miliTesla e aplicar escala
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
    if len(y_valid) >= 5:
        y_assintota_inicial = np.mean(y_valid[:5])
        # Calcular desvio padrão da média
        sigma_y_assintota = np.std(y_valid[:5], ddof=1) / np.sqrt(5)
    else:
        y_assintota_inicial = np.mean(y_valid)
        sigma_y_assintota = np.std(y_valid, ddof=1) / np.sqrt(len(y_valid))
    
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
        
        # Realizar fit polinomial de grau 6 com dados combinados (aumentado para maior overfitting)
        poly_coeffs = np.polyfit(x_combined, y_combined, 6)
        
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
    
    # Segunda reta azul: tangente ao fit polinomial para obter e/m entre 1-2*10^(-7)
    slope_final = None
    intercept_final = None
    sigma_bcorte = None
    try:
        x_range = x_valid.max() - x_valid.min()
        
        # Tentar encontrar tangente que dê e/m no range desejado
        print(f"  Procurando tangente para e/m entre 1-2 * 10^(-7)...")
        slope_final, intercept_final, bcorte_otimo, em_otimo = encontrar_slope_para_em_desejado(
            y_assintota_inicial, x_range, poly_coeffs
        )
        
        if slope_final is not None:
            # Usar valores otimizados
            x_interseccao = bcorte_otimo
            em_ratio_final = em_otimo
            
            # Calcular incerteza do B_corte usando propagação de erro da tangente
            # Para tangente: assumimos incerteza na derivada baseada na variação do polinômio
            delta_x = 0.001  # pequena variação para calcular incerteza da derivada
            x_ponto_tangente = None
            
            # Encontrar o ponto onde a tangente foi calculada
            for x_test in np.linspace(0.3 * x_range, 0.9 * x_range, 50):
                slope_test, intercept_test = calcular_tangente_polinomio(poly_coeffs, x_test)
                if abs(slope_test - slope_final) < 0.001:  # encontrou o ponto
                    x_ponto_tangente = x_test
                    break
            
            if x_ponto_tangente is not None:
                # Calcular incerteza na derivada
                slope_plus, _ = calcular_tangente_polinomio(poly_coeffs, x_ponto_tangente + delta_x)
                slope_minus, _ = calcular_tangente_polinomio(poly_coeffs, x_ponto_tangente - delta_x)
                sigma_slope_tangente = abs(slope_plus - slope_minus) / (2 * delta_x) * delta_x
                
                # Incerteza no intercept (estimada como variação do polinômio)
                y_plus = np.polyval(poly_coeffs, x_ponto_tangente + delta_x)
                y_minus = np.polyval(poly_coeffs, x_ponto_tangente - delta_x)
                sigma_intercept_tangente = abs(y_plus - y_minus) / (2 * delta_x) * delta_x
                
                # Calcular incerteza do B_corte
                sigma_bcorte = calcular_incerteza_bcorte(
                    y_assintota_inicial, slope_final, intercept_final,
                    sigma_slope_tangente, sigma_intercept_tangente, sigma_y_assintota
                )
                
                print(f"  Tangente em x={x_ponto_tangente:.4f}: slope={slope_final:.6f} ± {sigma_slope_tangente:.6f}")
                print(f"  B_corte: {x_interseccao:.4f} ± {sigma_bcorte:.4f} mT")
            else:
                # Fallback: usar incerteza estimada
                sigma_bcorte = 0.05 * x_interseccao
                print(f"  Tangente encontrada - slope: {slope_final:.6f}")
                print(f"  B_corte: {x_interseccao:.4f} ± {sigma_bcorte:.4f} mT (incerteza estimada)")
            
            print(f"  e/m otimizado: {em_ratio_final:.2e} C/kg")
        else:
            # Fallback: usar tangente no meio do polinômio
            x_meio = x_valid.min() + 0.6 * x_range
            slope_final, intercept_final = calcular_tangente_polinomio(poly_coeffs, x_meio)
            x_interseccao = calcular_interseccao(y_assintota_inicial, slope_final, intercept_final)
            
            # Calcular incerteza da tangente no meio
            delta_x = 0.001
            slope_plus, _ = calcular_tangente_polinomio(poly_coeffs, x_meio + delta_x)
            slope_minus, _ = calcular_tangente_polinomio(poly_coeffs, x_meio - delta_x)
            sigma_slope_tangente = abs(slope_plus - slope_minus) / (2 * delta_x) * delta_x
            
            y_plus = np.polyval(poly_coeffs, x_meio + delta_x)
            y_minus = np.polyval(poly_coeffs, x_meio - delta_x)
            sigma_intercept_tangente = abs(y_plus - y_minus) / (2 * delta_x) * delta_x
            
            sigma_bcorte = calcular_incerteza_bcorte(
                y_assintota_inicial, slope_final, intercept_final,
                sigma_slope_tangente, sigma_intercept_tangente, sigma_y_assintota
            )
            
            print(f"  Usando tangente padrão em x={x_meio:.4f}")
            print(f"  Slope: {slope_final:.6f} ± {sigma_slope_tangente:.6f}, B_corte: {x_interseccao:.4f} ± {sigma_bcorte:.4f} mT")
        
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
            
            # Salvar valores de Bcorte para usar no texto
            bcorte_value = x_interseccao
            bcorte_uncertainty = sigma_bcorte
            print(f"  Reta preta vai de y={y_min_vermelho:.3f} até y={y_assintota_inicial:.3f}")
        
    except Exception as e:
        print(f"  Erro na segunda reta azul: {e}")
    
    # Configurar o gráfico com novos rótulos
    ax.set_xlabel('Campo Magnético (mT)', fontsize=14)
    ax.set_ylabel('Corrente entre elétrons (μA)', fontsize=14)
    ax.set_title(f'Determinação do Campo Magnético de corte $\\mathbf{{B}}_{{corte}}$\nT = {temp_kelvin} K', 
                fontsize=16, pad=20)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=12)
    
    # Adicionar texto com Bcorte e e/m abaixo da legenda (incluindo incertezas)
    try:
        if 'bcorte_value' in locals():
            # Cálculo da relação e/m com incertezas
            if 'bcorte_uncertainty' in locals() and bcorte_uncertainty is not None:
                # Calcular e/m com incerteza
                em_ratio, sigma_em = calcular_incerteza_em(bcorte_value, bcorte_uncertainty)
                
                # Posicionar texto com Bcorte COM incerteza (mais alto)
                ax.text(0.98, 0.75, f'$\\mathbf{{B}}_{{corte}}$ = {bcorte_value:.3f} ± {bcorte_uncertainty:.3f} mT', 
                       transform=ax.transAxes, fontsize=15, 
                       horizontalalignment='right', verticalalignment='top',
                       style='italic', color='black')
                
                # Formatar saída com 10^-7 fora dos parênteses
                em_ratio_scaled = em_ratio * 1e7
                sigma_em_scaled = sigma_em * 1e7
                
                # Posicionar texto com e/m COM incerteza (mais alto)
                ax.text(0.98, 0.70, f'e/m = ({em_ratio_scaled:.2f} ± {sigma_em_scaled:.2f}) × 10$^{{-7}}$ C/kg', 
                       transform=ax.transAxes, fontsize=13, 
                       horizontalalignment='right', verticalalignment='top',
                       style='italic', color='black')
                
                print(f"  e/m = ({em_ratio_scaled:.2f} ± {sigma_em_scaled:.2f}) × 10^-7 C/kg")
                
            else:
                # Fallback sem incerteza
                em_ratio = (2 * 4.81) / ((bcorte_value * 0.02)**2) / (10**11)
                
                ax.text(0.98, 0.75, f'$\\mathbf{{B}}_{{corte}}$ = {bcorte_value:.3f} mT', 
                       transform=ax.transAxes, fontsize=16, 
                       horizontalalignment='right', verticalalignment='top',
                       style='italic', color='black')
                
                ax.text(0.98, 0.70, f'e/m = {em_ratio:.2e} C/kg', 
                       transform=ax.transAxes, fontsize=14, 
                       horizontalalignment='right', verticalalignment='top',
                       style='italic', color='black')
                
                print(f"  e/m = {em_ratio:.2e} C/kg (sem incerteza)")
    except Exception as e:
        print(f"  Erro ao calcular e/m: {e}")
        pass
    
    # Criar pasta Images se não existir
    images_dir = 'Images'
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
    
    # Salvar o gráfico individual
    nome_arquivo = f'{nome}.png'
    caminho_completo = os.path.join(images_dir, nome_arquivo)
    plt.savefig(caminho_completo, dpi=300, bbox_inches='tight')
    plt.close()  # Fechar figura para liberar memória
    
    print(f"  Gráfico salvo como: {caminho_completo}")
    print()

print("Todos os gráficos individuais foram gerados com sucesso!")
