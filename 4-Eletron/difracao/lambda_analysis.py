"""
lambda_analysis.py

Análise dos comprimentos de onda de de Broglie calculados por dois métodos:
1. λ_tensão = √(150/V) - usando a tensão de aceleração
2. λ_difração = r/(d×L) - usando medidas de difração

Compara os resultados e gera tabela formatada com incertezas.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# =========================
# CONSTANTES EXPERIMENTAIS
# =========================

# Parâmetros do experimento
L = 0.135  # Distância em metros
L_uncertainty = 0.001  # Incerteza em L (m)

# Calibração de pixels para centímetros (da memória do experimento)
PIXELS_PER_CM = 80  # pixels/cm

# Parâmetros dos planos cristalinos
d1 = 2.13  # Angstrom - para r (raio menor)
d2 = 1.23  # Angstrom - para R (raio maior)  
d1_uncertainty = 0.01  # Angstrom
d2_uncertainty = 0.01  # Angstrom

# Incerteza da tensão
V_uncertainty = 0.1  # Volts

def format_uncertainty(value, uncertainty):
    """
    Formata valor e incerteza seguindo as regras de algarismos significativos.
    
    Regra: Arredonda incerteza para 1 algarismo significativo, 
           depois arredonda valor para mesma precisão.
    
    Exemplo: 1.5349 ± 0.0157 → 1.53 ± 0.02
    
    Returns:
        tuple: (valor_formatado, incerteza_formatada)
    """
    if uncertainty == 0 or np.isnan(uncertainty):
        return f"{value:.3f}", "0.000"
    
    # Encontra ordem de magnitude da incerteza
    if uncertainty > 0:
        magnitude = 10 ** np.floor(np.log10(uncertainty))
    else:
        magnitude = 1
    
    # Arredonda incerteza para 1 algarismo significativo
    uncertainty_rounded = np.round(uncertainty / magnitude) * magnitude
    
    # Determina número de casas decimais
    if magnitude >= 1:
        decimals = 0
    else:
        decimals = int(-np.floor(np.log10(magnitude)))
    
    # Arredonda valor para a mesma precisão da incerteza
    value_rounded = np.round(value / magnitude) * magnitude
    
    # Formata strings
    if decimals == 0:
        value_str = f"{value_rounded:.0f}"
        uncertainty_str = f"{uncertainty_rounded:.0f}"
    else:
        value_str = f"{value_rounded:.{decimals}f}"
        uncertainty_str = f"{uncertainty_rounded:.{decimals}f}"
    
    return value_str, uncertainty_str

def calculate_lambda_tension(V_volts, debug=False):
    """
    Calcula λ de de Broglie usando a tensão de aceleração.
    
    Fórmula: λ = √(150/V) Angstrom
    
    Args:
        V_volts: Tensão em Volts
        debug: se True, imprime valores intermediários
    
    Returns:
        float: λ_tensão em Angstrom
    """
    lambda_tension = np.sqrt(150.0 / V_volts)
    
    if debug:
        print(f"\n=== DEBUG TENSÃO ===")
        print(f"V_volts: {V_volts:.1f} V")
        print(f"150/V: {150.0/V_volts:.3f}")
        print(f"λ_tensão: {lambda_tension:.6f} Å")
        print(f"====================")
    
    return lambda_tension

def calculate_lambda_tension_uncertainty(lambda_tension, V_volts):
    """
    Calcula incerteza de λ_tensão usando propagação de erros.
    
    σλ = λ × σV/(2V)
    
    Args:
        lambda_tension: valor de λ_tensão
        V_volts: tensão em Volts
    
    Returns:
        float: incerteza de λ_tensão
    """
    return lambda_tension * V_uncertainty / (2 * V_volts)

def calculate_lambda_diffraction(r_value_pixels, d_value, debug=False):
    """
    Calcula λ de de Broglie usando difração.
    
    Fórmula: λ = r×d/L onde r está em cm
    
    Args:
        r_value_pixels: raio em cm (já processado com fator 4.656)
        d_value: espaçamento cristalino em Angstrom
        debug: se True, imprime valores intermediários
    
    Returns:
        float: λ_difração em Angstrom
    """
    # CONVERSÃO CORRETA: r já está em cm (com fator 4.656 aplicado), converte para metros
    r_meters = r_value_pixels / 100.0  # cm -> metros
    
    if debug:
        print(f"\n=== DEBUG DIFRAÇÃO (CORRIGIDO) ===")
        print(f"r_cm: {r_value_pixels:.3f} cm (já com fator 4.656 aplicado)")
        print(f"r_metros: {r_meters:.9f} m")
        print(f"d_angstrom: {d_value:.2f} Å")
        print(f"L: {L:.3f} m")
        
    # Fórmula correta: λ = r×d/L
    # r em metros, d em Angstrom, L em metros -> resultado direto em Angstrom
    lambda_angstrom = (r_meters * d_value) / L
    
    if debug:
        print(f"λ_angstrom: {lambda_angstrom:.6f} Å")
        print(f"=====================")
    
    return lambda_angstrom

def calculate_lambda_diffraction_uncertainty(lambda_diffraction, r_value_pixels, r_uncertainty_pixels, d_value, d_uncertainty):
    """
    Calcula incerteza de λ_difração usando propagação de erros.
    
    σλ = λ × √[(σr/r)² + (σd/d)² + (σL/L)²]
    
    Args:
        lambda_diffraction: valor de λ_difração (pode ser Series)
        r_value_pixels: raio em pixels ajustados (pode ser Series)
        r_uncertainty_pixels: incerteza do raio em pixels (delta_r do processed.csv, pode ser Series)
        d_value: espaçamento cristalino em Angstrom (escalar)
        d_uncertainty: incerteza de d (escalar)
    
    Returns:
        array: incerteza de λ_difração
    """
    # Termos relativos (usando np.where para lidar com Series)
    # A conversão de pixels para metros cancela na razão σr/r
    term_r = np.where(r_value_pixels != 0, r_uncertainty_pixels / r_value_pixels, 0)
    term_d = d_uncertainty / d_value if d_value != 0 else 0
    term_L = L_uncertainty / L if L != 0 else 0
    
    # Propagação de erro
    relative_uncertainty = np.sqrt(term_r**2 + term_d**2 + term_L**2)
    
    return lambda_diffraction * relative_uncertainty

def process_lambda_data():
    """
    Processa dados do processed.csv e calcula comprimentos de onda pelos dois métodos.
    
    Returns:
        DataFrame: tabela com todos os cálculos
    """
    # Carrega dados processados
    data_path = "Data/processed.csv"
    df = pd.read_csv(data_path)
    
    # Converte voltagem de 15-50V para 1.5-5.0V
    df['V_real'] = df['Volts'] / 10.0
    
    # FATOR DE CORREÇÃO IDENTIFICADO PELO USUÁRIO
    # Multiplica todos os raios e incertezas por 4.656
    correction_factor = 4.656
    print(f"Aplicando fator de correção: {correction_factor}")
    
    df['r'] = df['r'] * correction_factor
    df['R'] = df['R'] * correction_factor  
    df['delta_r'] = df['delta_r'] * correction_factor
    df['delta_R'] = df['delta_R'] * correction_factor
    
    # Debug: analisa primeiro ponto detalhadamente
    print(f"\n=== ANÁLISE PRIMEIRO PONTO (V={df['V_real'].iloc[0]:.1f}V) ===")
    
    # Calcula λ_tensão
    first_lambda_tension = calculate_lambda_tension(df['V_real'].iloc[0], debug=True)
    df['lambda_tension'] = calculate_lambda_tension(df['V_real'])
    df['lambda_tension_uncertainty'] = calculate_lambda_tension_uncertainty(
        df['lambda_tension'], df['V_real']
    )
    
    # Calcula λ_difração para d1 (raio menor r)
    first_lambda_d1 = calculate_lambda_diffraction(df['r'].iloc[0], d1, debug=True)
    df['lambda_d1'] = calculate_lambda_diffraction(df['r'], d1)
    df['lambda_d1_uncertainty'] = calculate_lambda_diffraction_uncertainty(
        df['lambda_d1'], df['r'], df['delta_r'], d1, d1_uncertainty
    )
    
    # Calcula λ_difração para d2 (raio maior R)  
    first_lambda_d2 = calculate_lambda_diffraction(df['R'].iloc[0], d2, debug=True)
    df['lambda_d2'] = calculate_lambda_diffraction(df['R'], d2)
    df['lambda_d2_uncertainty'] = calculate_lambda_diffraction_uncertainty(
        df['lambda_d2'], df['R'], df['delta_R'], d2, d2_uncertainty
    )
    
    print(f"\n=== ANÁLISE DE FATORES DE ERRO ===")
    print(f"λ_tensão:  {first_lambda_tension:.6f} Å")
    print(f"λ_d1:      {first_lambda_d1:.6f} Å") 
    print(f"λ_d2:      {first_lambda_d2:.6f} Å")
    
    # Calcula fatores de erro
    fator_d1 = first_lambda_tension / first_lambda_d1  # Quanto precisa multiplicar d1 para igualar tensão
    fator_d2 = first_lambda_tension / first_lambda_d2  # Quanto precisa multiplicar d2 para igualar tensão
    
    print(f"\nFATORES DE CORREÇÃO NECESSÁRIOS:")
    print(f"Para λ_d1 igualar λ_tensão: multiplicar por {fator_d1:.3f}")
    print(f"Para λ_d2 igualar λ_tensão: multiplicar por {fator_d2:.3f}")
    print(f"Diferença entre fatores: {abs(fator_d1-fator_d2):.3f}")
    
    if abs(fator_d1 - fator_d2) < 0.5:  # Se fatores são próximos
        fator_medio = (fator_d1 + fator_d2) / 2
        print(f"\n✓ FATORES SÃO PRÓXIMOS! Fator médio: {fator_medio:.3f}")
        print(f"Erro é aproximadamente LINEAR - pode ser corrigido")
    else:
        print(f"\n✗ FATORES SÃO DIFERENTES! Erro pode não ser linear")
    
    # Verifica se o fator é constante para várias tensões
    print(f"\n=== VERIFICAÇÃO DE CONSISTÊNCIA ===")
    print("Testando fatores para diferentes tensões:")
    
    for i in range(0, min(len(df), 10), 2):  # Testa algumas tensões
        V = df['V_real'].iloc[i]
        lambda_t = calculate_lambda_tension(V)
        lambda_d1 = calculate_lambda_diffraction(df['r'].iloc[i], d1)
        lambda_d2 = calculate_lambda_diffraction(df['R'].iloc[i], d2) 
        
        f1 = lambda_t / lambda_d1
        f2 = lambda_t / lambda_d2
        
        print(f"V={V:.1f}V: fator_d1={f1:.3f}, fator_d2={f2:.3f}")
    
    print("========================================")
    
    # Análise dimensional com fator de correção aplicado
    print(f"\n=== ANÁLISE DIMENSIONAL (CORRIGIDA) ===")
    print(f"r corrigido: {df['r'].iloc[0]:.3f} cm (4.656x aplicado)")  
    r_corr_meters = df['r'].iloc[0]/100  # cm -> m
    print(f"r em metros: {r_corr_meters:.2e} m")
    print(f"d: {d1:.2f} Å = {d1*1e-10:.2e} m")
    print(f"L: {L:.3f} m")
    print(f"Fórmula: λ = r × d / L")
    print(f"λ esperado = {r_corr_meters:.2e} × {d1:.2f} / {L:.3f} = {(r_corr_meters*d1)/L:.2f} Å")
    print("========================================")
    
    return df

def create_lambda_table(df, output_dir="Graficos"):
    """
    Cria tabela formatada no estilo da referência com valores e incertezas.
    """
    # Cria diretório se não existir
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Prepara dados da tabela
    table_data = []
    
    for _, row in df.iterrows():
        V = row['V_real']
        
        # Formata λ_tensão com incerteza
        lambda_t_str, lambda_t_unc_str = format_uncertainty(
            row['lambda_tension'], row['lambda_tension_uncertainty']
        )
        
        # Formata λ_d1 com incerteza
        lambda_d1_str, lambda_d1_unc_str = format_uncertainty(
            row['lambda_d1'], row['lambda_d1_uncertainty']
        )
        
        # Formata λ_d2 com incerteza  
        lambda_d2_str, lambda_d2_unc_str = format_uncertainty(
            row['lambda_d2'], row['lambda_d2_uncertainty']
        )
        
        table_data.append([
            f"{V:.1f}",  # Voltagem
            f"{lambda_t_str} ± {lambda_t_unc_str}",  # λ_tensão
            f"{lambda_d1_str} ± {lambda_d1_unc_str}",  # λ_d1
            f"{lambda_d2_str} ± {lambda_d2_unc_str}"   # λ_d2
        ])
    
    # Cria figura para tabela
    n_rows = len(table_data)
    fig, ax = plt.subplots(figsize=(12, n_rows * 0.4 + 3))
    ax.axis('tight')
    ax.axis('off')
    
    # Título com mais espaçamento
    title_text = "Comprimentos de Onda de de Broglie - Comparação de Métodos"
    ax.set_title(title_text, fontsize=16, weight='bold', pad=40)
    
    # Cabeçalho com unidades
    headers = [
        "V (V)", 
        "λ_tensão (Å)", 
        "λ_d₁ (Å)", 
        "λ_d₂ (Å)"
    ]
    
    # Cria tabela
    table = ax.table(
        cellText=table_data,
        colLabels=headers,
        cellLoc='center',
        loc='center',
        colWidths=[0.15, 0.28, 0.28, 0.28]
    )
    
    # Formatação da tabela
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.2)
    
    # Estiliza cabeçalho (azul como na referência)
    for i in range(4):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white', fontsize=12)
    
    # Adiciona informações dos parâmetros experimentais
    params_text = (
        f"Parâmetros experimentais:\n"
        f"L = ({L:.3f} ± {L_uncertainty:.3f}) m\n"
        f"d₁ = ({d1:.2f} ± {d1_uncertainty:.2f}) Å\n" 
        f"d₂ = ({d2:.2f} ± {d2_uncertainty:.2f}) Å\n"
        f"σ_V = {V_uncertainty:.1f} V"
    )
    
    ax.text(0.5, -0.15, params_text, ha='center', fontsize=11,
            style='italic', verticalalignment='top', transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7))
    
    # Salva tabela
    output_file = output_path / "tabela_comprimentos_onda.png"
    fig.tight_layout()
    fig.savefig(output_file, dpi=200, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Tabela salva em: {output_file}")
    
    return table_data

def print_summary_statistics(df):
    """
    Imprime estatísticas resumo dos cálculos.
    """
    print("\n" + "="*60)
    print("RESUMO ESTATÍSTICO - COMPRIMENTOS DE ONDA")
    print("="*60)
    
    print(f"\nFaixa de tensão analisada: {df['V_real'].min():.1f} - {df['V_real'].max():.1f} V")
    print(f"Número de medidas: {len(df)}")
    
    print(f"\nλ_tensão:")
    print(f"  Média: {df['lambda_tension'].mean():.3f} ± {df['lambda_tension_uncertainty'].mean():.3f} Å")
    print(f"  Faixa: {df['lambda_tension'].min():.3f} - {df['lambda_tension'].max():.3f} Å")
    
    print(f"\nλ_d1 (d = {d1} Å):")  
    print(f"  Média: {df['lambda_d1'].mean():.3f} ± {df['lambda_d1_uncertainty'].mean():.3f} Å")
    print(f"  Faixa: {df['lambda_d1'].min():.3f} - {df['lambda_d1'].max():.3f} Å")
    
    print(f"\nλ_d2 (d = {d2} Å):")
    print(f"  Média: {df['lambda_d2'].mean():.3f} ± {df['lambda_d2_uncertainty'].mean():.3f} Å") 
    print(f"  Faixa: {df['lambda_d2'].min():.3f} - {df['lambda_d2'].max():.3f} Å")
    
    print("\n" + "="*60)

def main():
    """
    Função principal - executa análise completa dos comprimentos de onda.
    """
    print("Iniciando análise dos comprimentos de onda de de Broglie...")
    
    # Verifica se arquivo processado existe
    if not Path("Data/processed.csv").exists():
        print("Erro: Arquivo Data/processed.csv não encontrado!")
        print("Execute primeiro o script process_data.py")
        return
    
    # Processa dados
    print("\nCarregando e processando dados...")
    df = process_lambda_data()
    
    # Mostra estatísticas
    print_summary_statistics(df)
    
    # Gera tabela  
    print("\nGerando tabela formatada...")
    table_data = create_lambda_table(df)
    
    print(f"\nAnálise concluída!")
    print(f"- {len(df)} medidas processadas")
    print(f"- 3 comprimentos de onda calculados para cada tensão")
    print(f"- Tabela salva com incertezas formatadas")
    
    # Salva dados detalhados em CSV para referência
    output_csv = "Data/lambda_analysis_detailed.csv" 
    df.to_csv(output_csv, index=False)
    print(f"- Dados detalhados salvos em: {output_csv}")
    
    return df

if __name__ == "__main__":
    result = main()
