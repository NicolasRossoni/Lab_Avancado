#!/usr/bin/env python3
"""
Análise dos comprimentos de onda de de Broglie - Dados do Paquímetro
Experimento 4-Eletron

Compara dois métodos para calcular λ de de Broglie:
1. λ_tensão = √(150/V) Å - método analítico baseado na tensão
2. λ_difração = r×d/L - método baseado nas medidas de difração com paquímetro

Os dados foram medidos manualmente com paquímetro (mais confiáveis).
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle

# Parâmetros experimentais
L = 0.135  # Distância em metros (± 0.001 m)
L_uncertainty = 0.001

# Espaçamentos cristalinos em Angstrom
d1 = 2.13  # ± 0.01 Å (para raio menor - d1)
d1_uncertainty = 0.01
d2 = 1.23  # ± 0.01 Å (para raio maior - d2) 
d2_uncertainty = 0.01

# Incerteza da tensão
V_uncertainty = 0.1  # V

def calculate_lambda_tension(V_volts, debug=False):
    """
    Calcula λ de de Broglie a partir da tensão.
    
    Fórmula: λ = √(150/V) Å
    
    Args:
        V_volts: tensão em Volts
        debug: se True, imprime valores intermediários
    
    Returns:
        float: λ_tensão em Angstrom
    """
    lambda_tension = np.sqrt(150.0 / V_volts)
    
    if debug:
        print(f"\n=== DEBUG TENSÃO ===")
        print(f"V_volts: {V_volts} V")
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

def calculate_lambda_diffraction(r_value_cm, d_value, debug=False):
    """
    Calcula λ de de Broglie usando difração.
    
    Fórmula: λ = r×d/L onde r está em cm
    
    Args:
        r_value_cm: raio em cm (dados do paquímetro)
        d_value: espaçamento cristalino em Angstrom
        debug: se True, imprime valores intermediários
    
    Returns:
        float: λ_difração em Angstrom
    """
    # Conversão: r está em cm, converte para metros
    r_meters = r_value_cm / 100.0  # cm -> metros
    
    if debug:
        print(f"\n=== DEBUG DIFRAÇÃO (PAQUÍMETRO) ===")
        print(f"r_cm: {r_value_cm:.3f} cm (medido com paquímetro)")
        print(f"r_metros: {r_meters:.9f} m")
        print(f"d_angstrom: {d_value:.2f} Å")
        print(f"L: {L:.3f} m")
        
    # Fórmula: λ = r×d/L
    # r em metros, d em Angstrom, L em metros -> resultado direto em Angstrom
    lambda_angstrom = (r_meters * d_value) / L
    
    if debug:
        print(f"λ_angstrom: {lambda_angstrom:.6f} Å")
        print(f"=====================")
    
    return lambda_angstrom

def calculate_lambda_diffraction_uncertainty(lambda_diffraction, r_value_cm, r_uncertainty_cm, d_value, d_uncertainty):
    """
    Calcula incerteza de λ_difração usando propagação de erros.
    
    σλ = λ × √[(σr/r)² + (σd/d)² + (σL/L)²]
    
    Args:
        lambda_diffraction: valor de λ_difração
        r_value_cm: valor do raio em cm
        r_uncertainty_cm: incerteza do raio em cm
        d_value: espaçamento cristalino em Angstrom
        d_uncertainty: incerteza do espaçamento cristalino
    
    Returns:
        float: incerteza de λ_difração
    """
    # Termos da propagação de erro
    sigma_r_over_r = r_uncertainty_cm / r_value_cm
    sigma_d_over_d = d_uncertainty / d_value
    sigma_L_over_L = L_uncertainty / L
    
    # Propagação de erro: σλ = λ × √[(σr/r)² + (σd/d)² + (σL/L)²]
    uncertainty = lambda_diffraction * np.sqrt(
        sigma_r_over_r**2 + sigma_d_over_d**2 + sigma_L_over_L**2
    )
    
    return uncertainty

def load_and_process_paquimetro_data():
    """
    Carrega e processa dados do paquímetro dos arquivos CSV.
    
    Returns:
        DataFrame: tabela com todos os cálculos
    """
    # Carrega dados de d1 (raio menor)
    df_d1 = pd.read_csv("Data/paquimetro_d1.csv")
    df_d2 = pd.read_csv("Data/paquimetro_d2.csv")
    
    print("Carregando e processando dados do paquímetro...")
    
    # Converte vírgulas para pontos e transforma em float
    for col in ['diametro1', 'diametro2']:
        df_d1[col] = df_d1[col].str.replace(',', '.').astype(float)
        df_d2[col] = df_d2[col].str.replace(',', '.').astype(float)
    
    # Converte tensão para float
    df_d1['Tensão (V)'] = df_d1['Tensão (V)'].str.replace(',', '.').astype(float)
    df_d2['Tensão (V)'] = df_d2['Tensão (V)'].str.replace(',', '.').astype(float)
    
    # Cria DataFrame combinado
    df = pd.DataFrame()
    df['V_real'] = df_d1['Tensão (V)']  # Tensão já está correta
    
    # Calcula raios (diâmetro/2) em cm
    df['r'] = (df_d1['diametro1'] + df_d1['diametro2']) / 4.0  # Média dos raios de d1
    df['R'] = (df_d2['diametro1'] + df_d2['diametro2']) / 4.0  # Média dos raios de d2
    
    # Calcula incertezas dos raios (metade da diferença dos diâmetros)
    df['delta_r'] = abs(df_d1['diametro1'] - df_d1['diametro2']) / 4.0
    df['delta_R'] = abs(df_d2['diametro1'] - df_d2['diametro2']) / 4.0
    
    print(f"Processados {len(df)} pontos de tensão: {df['V_real'].min():.1f} - {df['V_real'].max():.1f} V")
    
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
    
    print(f"\n=== VERIFICAÇÃO DETALHADA DAS FÓRMULAS ===")
    V_test = df['V_real'].iloc[0]
    r_test = df['r'].iloc[0] 
    R_test = df['R'].iloc[0]
    
    print(f"DADOS DE ENTRADA (V={V_test}V):")
    print(f"r = {r_test:.3f} cm = {r_test/100:.6f} m")
    print(f"R = {R_test:.3f} cm = {R_test/100:.6f} m")
    print(f"d1 = {d1:.2f} Å, d2 = {d2:.2f} Å")
    print(f"L = {L:.3f} m")
    
    print(f"\nFÓRMULA λ_tensão = √(150/V):")
    print(f"λ_tensão = √(150/{V_test}) = √{150/V_test:.1f} = {first_lambda_tension:.6f} Å")
    
    print(f"\nFÓRMULA λ_difração = r×d/L:")
    print(f"λ_d1 = ({r_test/100:.6f} × {d1:.2f}) / {L:.3f}")
    print(f"λ_d1 = {(r_test/100)*d1:.8f} / {L:.3f} = {first_lambda_d1:.6f} Å")
    print(f"λ_d2 = ({R_test/100:.6f} × {d2:.2f}) / {L:.3f}")
    print(f"λ_d2 = {(R_test/100)*d2:.8f} / {L:.3f} = {first_lambda_d2:.6f} Å")
    
    print(f"\n=== ANÁLISE DE FATORES DE ERRO (PAQUÍMETRO) ===")
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
        
        # Testa se aplicando o fator corrige os valores
        print(f"\nTESTE: Aplicando fator médio {fator_medio:.3f}:")
        lambda_d1_corrigido = first_lambda_d1 * fator_medio
        lambda_d2_corrigido = first_lambda_d2 * fator_medio
        print(f"λ_d1 corrigido: {lambda_d1_corrigido:.3f} Å (vs λ_tensão: {first_lambda_tension:.3f} Å)")
        print(f"λ_d2 corrigido: {lambda_d2_corrigido:.3f} Å (vs λ_tensão: {first_lambda_tension:.3f} Å)")
    else:
        print(f"\n✗ FATORES SÃO DIFERENTES! Erro pode não ser linear")
    
    # Verifica se o fator é constante para várias tensões
    print(f"\n=== VERIFICAÇÃO DE CONSISTÊNCIA (PAQUÍMETRO) ===")
    print("Testando fatores para diferentes tensões:")
    
    for i in range(len(df)):  # Testa todas as tensões disponíveis
        V = df['V_real'].iloc[i]
        lambda_t = calculate_lambda_tension(V)
        lambda_d1 = calculate_lambda_diffraction(df['r'].iloc[i], d1)
        lambda_d2 = calculate_lambda_diffraction(df['R'].iloc[i], d2) 
        
        f1 = lambda_t / lambda_d1
        f2 = lambda_t / lambda_d2
        
        print(f"V={V:.1f}V: fator_d1={f1:.3f}, fator_d2={f2:.3f}")
    
    print("==========================================")
    
    # Análise dimensional 
    print(f"\n=== ANÁLISE DIMENSIONAL (PAQUÍMETRO) ===")
    print(f"r medido: {df['r'].iloc[0]:.3f} cm (paquímetro)")  
    r_meters = df['r'].iloc[0]/100  # cm -> m
    print(f"r em metros: {r_meters:.2e} m")
    print(f"d: {d1:.2f} Å")
    print(f"L: {L:.3f} m")
    print(f"Fórmula: λ = r × d / L")
    print(f"λ esperado = {r_meters:.2e} × {d1:.2f} / {L:.3f} = {(r_meters*d1)/L:.2f} Å")
    print("========================================")
    
    return df

def round_to_significant_uncertainty(value, uncertainty):
    """
    Arredonda incerteza para 1 algarismo significativo e valor para a mesma precisão.
    """
    if uncertainty == 0:
        return value, uncertainty
    
    # Encontra a ordem de grandeza da incerteza
    uncertainty_magnitude = np.floor(np.log10(abs(uncertainty)))
    
    # Arredonda incerteza para 1 algarismo significativo
    uncertainty_rounded = np.round(uncertainty, -int(uncertainty_magnitude))
    
    # Arredonda valor para a mesma precisão
    value_rounded = np.round(value, -int(uncertainty_magnitude))
    
    # Determina quantas casas decimais mostrar
    decimal_places = max(0, -int(uncertainty_magnitude))
    
    return value_rounded, uncertainty_rounded, decimal_places

def create_lambda_table_paquimetro(df, output_dir="Graficos"):
    """
    Cria tabela formatada no estilo da referência com valores e incertezas - dados do paquímetro.
    """
    # Cria diretório se não existir
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Configurações da figura
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')
    
    # Título da tabela
    title = "Comprimentos de Onda de de Broglie - Dados do Paquímetro"
    ax.text(0.5, 0.95, title, ha='center', va='center', fontsize=16, fontweight='bold')
    
    # Cabeçalhos das colunas
    headers = ['V (V)', 'λ_tensão (Å)', 'λ_d1 (Å)', 'λ_d2 (Å)']
    col_widths = [0.15, 0.25, 0.25, 0.25]
    
    # Posições
    y_start = 0.85
    row_height = 0.08
    
    # Desenha cabeçalhos
    y_pos = y_start
    x_pos = 0.1
    for i, (header, width) in enumerate(zip(headers, col_widths)):
        rect = Rectangle((x_pos, y_pos-row_height/2), width, row_height, 
                        linewidth=1, edgecolor='black', facecolor='lightgray')
        ax.add_patch(rect)
        ax.text(x_pos + width/2, y_pos, header, ha='center', va='center', 
                fontweight='bold', fontsize=11)
        x_pos += width
    
    # Preenche dados
    for idx, row in df.iterrows():
        y_pos -= row_height
        x_pos = 0.1
        
        # Arredonda valores com incertezas corretamente
        tension_val, tension_unc, tension_dec = round_to_significant_uncertainty(
            row['lambda_tension'], row['lambda_tension_uncertainty'])
        d1_val, d1_unc, d1_dec = round_to_significant_uncertainty(
            row['lambda_d1'], row['lambda_d1_uncertainty'])
        d2_val, d2_unc, d2_dec = round_to_significant_uncertainty(
            row['lambda_d2'], row['lambda_d2_uncertainty'])
        
        # Formata strings com precisão correta
        tension_str = f"{tension_val:.{tension_dec}f} ± {tension_unc:.{tension_dec}f}"
        d1_str = f"{d1_val:.{d1_dec}f} ± {d1_unc:.{d1_dec}f}"
        d2_str = f"{d2_val:.{d2_dec}f} ± {d2_unc:.{d2_dec}f}"
        
        values = [f"{row['V_real']:.1f}", tension_str, d1_str, d2_str]
        
        for i, (value, width) in enumerate(zip(values, col_widths)):
            rect = Rectangle((x_pos, y_pos-row_height/2), width, row_height, 
                           linewidth=1, edgecolor='black', facecolor='white')
            ax.add_patch(rect)
            ax.text(x_pos + width/2, y_pos, value, ha='center', va='center', 
                    fontsize=10)
            x_pos += width
    
    # Rodapé com informações experimentais
    footer_y = y_pos - 0.15
    footer_text = f"Parâmetros: L = {L:.3f} ± {L_uncertainty:.3f} m, d₁ = {d1:.2f} ± {d1_uncertainty:.2f} Å, d₂ = {d2:.2f} ± {d2_uncertainty:.2f} Å\nDados medidos com paquímetro (manual)"
    ax.text(0.5, footer_y, footer_text, ha='center', va='center', 
            fontsize=9, style='italic')
    
    # Salva tabela
    plt.tight_layout()
    table_path = output_path / "tabela_comprimentos_onda_paquimetro.png"
    plt.savefig(table_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Tabela salva em: {table_path}")

def generate_summary_statistics(df):
    """
    Gera resumo estatístico dos dados.
    """
    print("\n" + "="*60)
    print("RESUMO ESTATÍSTICO - COMPRIMENTOS DE ONDA (PAQUÍMETRO)")
    print("="*60)
    
    print(f"\nFaixa de tensão analisada: {df['V_real'].min():.1f} - {df['V_real'].max():.1f} V")
    print(f"Número de medidas: {len(df)}")
    
    # Estatísticas para cada lambda
    for col_name, display_name in [
        ('lambda_tension', 'λ_tensão'),
        ('lambda_d1', 'λ_d1 (d = 2.13 Å)'),
        ('lambda_d2', 'λ_d2 (d = 1.23 Å)')
    ]:
        mean_val = df[col_name].mean()
        std_val = df[col_name].std()
        min_val = df[col_name].min()
        max_val = df[col_name].max()
        
        print(f"\n{display_name}:")
        print(f"  Média: {mean_val:.3f} ± {std_val:.3f} Å")
        print(f"  Faixa: {min_val:.3f} - {max_val:.3f} Å")
    
    print("\n" + "="*60)

def main():
    """
    Função principal que executa toda a análise.
    """
    print("Iniciando análise dos comprimentos de onda de de Broglie...")
    print("Dados obtidos com paquímetro (medição manual)")
    print()
    
    # Carrega e processa dados
    df = load_and_process_paquimetro_data()
    
    # Gera resumo estatístico
    generate_summary_statistics(df)
    
    # Gera tabela formatada
    print("\nGerando tabela formatada...")
    create_lambda_table_paquimetro(df)
    
    # Salva dados detalhados
    output_csv = "Data/lambda_analysis_paquimetro_detailed.csv"
    df.to_csv(output_csv, index=False, float_format='%.6f')
    print(f"Dados detalhados salvos em: {output_csv}")
    
    print("\nAnálise concluída!")
    print(f"- {len(df)} medidas processadas")
    print("- 3 comprimentos de onda calculados para cada tensão")
    print("- Tabela salva com incertezas formatadas")
    print("- Dados medidos com paquímetro (alta confiabilidade)")

if __name__ == "__main__":
    main()
