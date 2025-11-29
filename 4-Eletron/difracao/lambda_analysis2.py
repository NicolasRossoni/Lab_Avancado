"""
lambda_analysis2.py

Análise dos comprimentos de onda de de Broglie com dados corrigidos.
Gera tabelas separadas para:
1. Dados do paquímetro (finaldata_paquimetro.csv)
2. Dados computacionais (finaldata_computacional.csv)

Compara dois métodos para calcular λ:
- λ_tensão = √(150/V) - método analítico
- λ_difração = r×d/L - método baseado em difração
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# =========================
# CONSTANTES EXPERIMENTAIS
# =========================

# Parâmetros do experimento
L = 0.135 / 3  # Distância em metros (dividido por 3 conforme correção)
L_uncertainty = 0.001  # Incerteza em L (m) - mantém 1 mm

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

def calculate_lambda_tension(V_volts):
    """
    Calcula λ de de Broglie usando a tensão de aceleração.
    
    Fórmula: λ = √(150/V) Angstrom
    """
    return np.sqrt(150.0 / V_volts)

def calculate_lambda_tension_uncertainty(lambda_tension, V_volts):
    """
    Calcula incerteza de λ_tensão usando propagação de erros.
    
    σλ = λ × σV/(2V)
    """
    return lambda_tension * V_uncertainty / (2 * V_volts)

def calculate_lambda_diffraction(r_value_meters, d_value):
    """
    Calcula λ de de Broglie usando difração.
    
    Fórmula: λ = r×d/L
    
    Args:
        r_value_meters: raio em metros
        d_value: espaçamento cristalino em Angstrom
    
    Returns:
        float: λ_difração em Angstrom
    """
    # r em metros, d em Angstrom, L em metros -> resultado em Angstrom
    lambda_angstrom = (r_value_meters * d_value) / L
    return lambda_angstrom

def calculate_lambda_diffraction_uncertainty(lambda_diffraction, r_value_meters, r_uncertainty_meters, d_value, d_uncertainty):
    """
    Calcula incerteza de λ_difração usando propagação de erros.
    
    σλ = λ × √[(σr/r)² + (σd/d)² + (σL/L)²]
    """
    # Termos relativos
    term_r = np.where(r_value_meters != 0, r_uncertainty_meters / r_value_meters, 0)
    term_d = d_uncertainty / d_value if d_value != 0 else 0
    term_L = L_uncertainty / L if L != 0 else 0
    
    # Propagação de erro
    relative_uncertainty = np.sqrt(term_r**2 + term_d**2 + term_L**2)
    
    return lambda_diffraction * relative_uncertainty

def process_data(df, data_source_name):
    """
    Processa dados e calcula comprimentos de onda pelos dois métodos.
    
    Args:
        df: DataFrame com colunas [Volts, r, R, delta_r, delta_R] em metros
        data_source_name: nome da fonte de dados para debug
    
    Returns:
        DataFrame: tabela com todos os cálculos
    """
    print(f"\n{'='*60}")
    print(f"PROCESSANDO: {data_source_name}")
    print(f"{'='*60}")
    
    df_result = df.copy()
    
    # Calcula λ_tensão
    df_result['lambda_tension'] = calculate_lambda_tension(df_result['Volts'])
    df_result['lambda_tension_uncertainty'] = calculate_lambda_tension_uncertainty(
        df_result['lambda_tension'], df_result['Volts']
    )
    
    # Calcula λ_difração para d1 (raio menor r)
    df_result['lambda_d1'] = calculate_lambda_diffraction(df_result['r'], d1)
    df_result['lambda_d1_uncertainty'] = calculate_lambda_diffraction_uncertainty(
        df_result['lambda_d1'], df_result['r'], df_result['delta_r'], d1, d1_uncertainty
    )
    
    # Calcula λ_difração para d2 (raio maior R)  
    df_result['lambda_d2'] = calculate_lambda_diffraction(df_result['R'], d2)
    df_result['lambda_d2_uncertainty'] = calculate_lambda_diffraction_uncertainty(
        df_result['lambda_d2'], df_result['R'], df_result['delta_R'], d2, d2_uncertainty
    )
    
    # Estatísticas
    print(f"\n📊 Estatísticas de {data_source_name}:")
    print(f"  Voltagens: {df_result['Volts'].min():.1f} - {df_result['Volts'].max():.1f} V")
    print(f"  Número de medidas: {len(df_result)}")
    print(f"\n  λ_tensão:")
    print(f"    Média: {df_result['lambda_tension'].mean():.3f} Å")
    print(f"    Faixa: {df_result['lambda_tension'].min():.3f} - {df_result['lambda_tension'].max():.3f} Å")
    print(f"\n  λ_d1 (d={d1} Å):")
    print(f"    Média: {df_result['lambda_d1'].mean():.3f} Å")
    print(f"    Faixa: {df_result['lambda_d1'].min():.3f} - {df_result['lambda_d1'].max():.3f} Å")
    print(f"\n  λ_d2 (d={d2} Å):")
    print(f"    Média: {df_result['lambda_d2'].mean():.3f} Å")
    print(f"    Faixa: {df_result['lambda_d2'].min():.3f} - {df_result['lambda_d2'].max():.3f} Å")
    
    return df_result

def create_lambda_table(df, data_source_name, output_filename):
    """
    Cria tabela formatada com valores e incertezas.
    
    Args:
        df: DataFrame com os cálculos de lambda
        data_source_name: nome para o título da tabela
        output_filename: nome do arquivo de saída (sem extensão)
    """
    # Cria diretório se não existir
    output_path = Path("Graficos")
    output_path.mkdir(exist_ok=True)
    
    # Prepara dados da tabela
    table_data = []
    
    for _, row in df.iterrows():
        V = row['Volts']
        
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
    
    # Título
    title_text = f"Comprimentos de Onda de de Broglie\n{data_source_name}"
    ax.set_title(title_text, fontsize=16, weight='bold', pad=40)
    
    # Cabeçalho
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
    
    # Estiliza cabeçalho
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
    output_file = output_path / f"{output_filename}.png"
    fig.tight_layout()
    fig.savefig(output_file, dpi=200, bbox_inches='tight')
    plt.close(fig)
    
    print(f"\n✅ Tabela salva em: {output_file}")
    
    return table_data

def create_paquimetro_plot(df_paq):
    """
    Cria gráfico para dados do paquímetro com 3 curvas.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Configurações de estilo
    colors = {
        'd1': '#E74C3C',  # Vermelho
        'd2': '#3498DB',  # Azul
        'tension': '#2C3E50'  # Preto/cinza escuro
    }
    
    # 1. λ_tensão
    ax.errorbar(
        df_paq['Volts'], 
        df_paq['lambda_tension'],
        yerr=df_paq['lambda_tension_uncertainty'],
        color=colors['tension'],
        marker='o',
        markersize=8,
        linestyle='-',
        linewidth=2.5,
        capsize=5,
        capthick=2,
        label='λ_tensão',
        zorder=3
    )
    
    # 2. λ_d1
    ax.errorbar(
        df_paq['Volts'],
        df_paq['lambda_d1'],
        yerr=df_paq['lambda_d1_uncertainty'],
        color=colors['d1'],
        marker='s',
        markersize=8,
        linestyle='-',
        linewidth=2.5,
        capsize=5,
        capthick=2,
        label='λ_d₁ (difração)',
        zorder=2
    )
    
    # 3. λ_d2
    ax.errorbar(
        df_paq['Volts'],
        df_paq['lambda_d2'],
        yerr=df_paq['lambda_d2_uncertainty'],
        color=colors['d2'],
        marker='s',
        markersize=8,
        linestyle='-',
        linewidth=2.5,
        capsize=5,
        capthick=2,
        label='λ_d₂ (difração)',
        zorder=1
    )
    
    # Configurações do gráfico
    ax.set_xlabel('Tensão de Aceleração (V)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Comprimento de Onda de de Broglie λ (Å)', fontsize=14, fontweight='bold')
    ax.set_title('Comprimento de Onda de de Broglie - Medidas com Paquímetro',
                 fontsize=16, fontweight='bold', pad=20)
    
    # Grade
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=1)
    
    # Legenda
    ax.legend(loc='best', fontsize=12, framealpha=0.95, shadow=True)
    
    # Ajustar limites dos eixos
    ax.set_xlim(df_paq['Volts'].min() - 0.2, df_paq['Volts'].max() + 0.2)
    
    # Adicionar informações dos parâmetros
    info_text = (
        f'L = {L*1000:.1f} mm (± {L_uncertainty*1000:.1f} mm)\n'
        f'd₁ = {d1:.2f} Å (± {d1_uncertainty:.2f} Å)\n'
        f'd₂ = {d2:.2f} Å (± {d2_uncertainty:.2f} Å)'
    )
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Salvar
    output_path = Path('Graficos')
    output_path.mkdir(exist_ok=True)
    output_file = output_path / 'grafico_paquimetro.png'
    
    plt.tight_layout()
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"\n✅ Gráfico paquímetro salvo em: {output_file}")

def create_foto_plot(df_comp):
    """
    Cria gráfico para dados computacionais (foto) com 3 curvas.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Configurações de estilo
    colors = {
        'd1': '#E74C3C',  # Vermelho
        'd2': '#3498DB',  # Azul
        'tension': '#2C3E50'  # Preto/cinza escuro
    }
    
    # 1. λ_tensão
    ax.errorbar(
        df_comp['Volts'], 
        df_comp['lambda_tension'],
        yerr=df_comp['lambda_tension_uncertainty'],
        color=colors['tension'],
        marker='o',
        markersize=6,
        linestyle='-',
        linewidth=2,
        capsize=4,
        capthick=1.5,
        label='λ_tensão',
        zorder=3
    )
    
    # 2. λ_d1
    ax.errorbar(
        df_comp['Volts'],
        df_comp['lambda_d1'],
        yerr=df_comp['lambda_d1_uncertainty'],
        color=colors['d1'],
        marker='o',
        markersize=6,
        linestyle='-',
        linewidth=2,
        capsize=4,
        capthick=1.5,
        label='λ_d₁ (difração)',
        zorder=2
    )
    
    # 3. λ_d2
    ax.errorbar(
        df_comp['Volts'],
        df_comp['lambda_d2'],
        yerr=df_comp['lambda_d2_uncertainty'],
        color=colors['d2'],
        marker='o',
        markersize=6,
        linestyle='-',
        linewidth=2,
        capsize=4,
        capthick=1.5,
        label='λ_d₂ (difração)',
        zorder=1
    )
    
    # Configurações do gráfico
    ax.set_xlabel('Tensão de Aceleração (V)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Comprimento de Onda de de Broglie λ (Å)', fontsize=14, fontweight='bold')
    ax.set_title('Comprimento de Onda de de Broglie - Análise de Imagens (Computacional)',
                 fontsize=16, fontweight='bold', pad=20)
    
    # Grade
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=1)
    
    # Legenda
    ax.legend(loc='best', fontsize=12, framealpha=0.95, shadow=True)
    
    # Ajustar limites dos eixos
    ax.set_xlim(df_comp['Volts'].min() - 0.2, df_comp['Volts'].max() + 0.2)
    
    # Adicionar informações dos parâmetros
    info_text = (
        f'L = {L*1000:.1f} mm (± {L_uncertainty*1000:.1f} mm)\n'
        f'd₁ = {d1:.2f} Å (± {d1_uncertainty:.2f} Å)\n'
        f'd₂ = {d2:.2f} Å (± {d2_uncertainty:.2f} Å)'
    )
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Salvar
    output_path = Path('Graficos')
    output_path.mkdir(exist_ok=True)
    output_file = output_path / 'grafico_foto.png'
    
    plt.tight_layout()
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"\n✅ Gráfico foto salvo em: {output_file}")

def main():
    """
    Função principal - processa ambos os conjuntos de dados.
    """
    print("="*60)
    print("ANÁLISE DE COMPRIMENTOS DE ONDA - DADOS CORRIGIDOS")
    print("="*60)
    
    # Verificar se arquivos existem
    paq_path = Path("Data/finaldata_paquimetro.csv")
    comp_path = Path("Data/finaldata_computacional.csv")
    
    if not paq_path.exists():
        print(f"❌ Erro: {paq_path} não encontrado!")
        print("   Execute primeiro: prepare_final_data.py")
        return
    
    if not comp_path.exists():
        print(f"❌ Erro: {comp_path} não encontrado!")
        print("   Execute primeiro: prepare_final_data.py")
        return
    
    # 1. PROCESSAR DADOS DO PAQUÍMETRO
    print("\n" + "🔧"*30)
    df_paq = pd.read_csv(paq_path)
    df_paq_result = process_data(df_paq, "Dados do Paquímetro")
    create_lambda_table(df_paq_result, "Dados do Paquímetro", "tabela_lambda_paquimetro")
    
    # Salvar CSV detalhado
    csv_paq = "Data/lambda_analysis_paquimetro_final.csv"
    df_paq_result.to_csv(csv_paq, index=False)
    print(f"📄 CSV detalhado salvo em: {csv_paq}")
    
    # 2. PROCESSAR DADOS COMPUTACIONAIS
    print("\n" + "💻"*30)
    df_comp = pd.read_csv(comp_path)
    df_comp_result = process_data(df_comp, "Dados Computacionais")
    create_lambda_table(df_comp_result, "Dados Computacionais (Análise Interativa)", "tabela_lambda_computacional")
    
    # Salvar CSV detalhado
    csv_comp = "Data/lambda_analysis_computacional_final.csv"
    df_comp_result.to_csv(csv_comp, index=False)
    print(f"📄 CSV detalhado salvo em: {csv_comp}")
    
    # 3. GERAR GRÁFICOS SEPARADOS
    print("\n" + "📊"*30)
    print("\nGerando gráficos...")
    create_paquimetro_plot(df_paq_result)
    create_foto_plot(df_comp_result)
    
    # RESUMO FINAL
    print("\n" + "="*60)
    print("ANÁLISE CONCLUÍDA!")
    print("="*60)
    print("\n📊 Arquivos gerados:")
    print("  Tabelas:")
    print("    - Graficos/tabela_lambda_paquimetro.png")
    print("    - Graficos/tabela_lambda_computacional.png")
    print("  Gráficos:")
    print("    - Graficos/grafico_paquimetro.png")
    print("    - Graficos/grafico_foto.png")
    print("  CSVs detalhados:")
    print(f"    - {csv_paq}")
    print(f"    - {csv_comp}")
    
    print("\n📈 Resumo comparativo (L corrigido = {:.1f} mm):".format(L*1000))
    print(f"\n  PAQUÍMETRO ({len(df_paq_result)} medidas):")
    print(f"    λ_tensão médio: {df_paq_result['lambda_tension'].mean():.3f} Å")
    print(f"    λ_d1 médio:     {df_paq_result['lambda_d1'].mean():.3f} Å")
    print(f"    λ_d2 médio:     {df_paq_result['lambda_d2'].mean():.3f} Å")
    
    print(f"\n  COMPUTACIONAL ({len(df_comp_result)} medidas):")
    print(f"    λ_tensão médio: {df_comp_result['lambda_tension'].mean():.3f} Å")
    print(f"    λ_d1 médio:     {df_comp_result['lambda_d1'].mean():.3f} Å")
    print(f"    λ_d2 médio:     {df_comp_result['lambda_d2'].mean():.3f} Å")
    
    print("\n🎯 Verificação de concordância:")
    # Para voltagens em comum (3.0 a 5.0V)
    common_volts = df_paq_result['Volts'].values
    for v in common_volts:
        paq_d1 = df_paq_result[df_paq_result['Volts'] == v]['lambda_d1'].values[0]
        comp_d1 = df_comp_result[df_comp_result['Volts'] == v]['lambda_d1'].values[0]
        tension = df_paq_result[df_paq_result['Volts'] == v]['lambda_tension'].values[0]
        print(f"  {v:.1f}V: λ_tensão={tension:.3f} Å, λ_d1_paq={paq_d1:.3f} Å, λ_d1_foto={comp_d1:.3f} Å")
    
    print("\n" + "="*60)
    
    return df_paq_result, df_comp_result

if __name__ == "__main__":
    result_paq, result_comp = main()
