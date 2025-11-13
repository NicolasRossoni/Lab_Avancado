"""
em_analysis.py

Gera gráficos e tabelas para análise de e/m.
- Gráfico V_fixo: I vs R (2 curvas para V=61V e V=169.8V)
- Gráfico I_fixo: V vs R
- Tabelas com valores de e/m calculados
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Valor de referência de e/m
e_over_m_ref = 1.758820024e11  # C/kg (CODATA 2018)

# Constantes para cálculos teóricos
N = 154
r_bobina = 0.398  # metros
mu0 = 1.25663706212e-6
k = 0.716

def format_uncertainty(value, uncertainty):
    """
    Formata valor e incerteza seguindo as regras de algarismos significativos.
    
    Regra: Arredonda incerteza para 1 algarismo significativo, 
           depois arredonda valor para mesma precisão.
    """
    if uncertainty == 0 or np.isnan(uncertainty):
        return f"{value:.3e}", "0"
    
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

def create_combined_table():
    """
    Cria tabela combinada com resultados de V_fixo e I_fixo.
    """
    # Carregar dados processados
    df_V = pd.read_csv("Data/processed_V_fixo.csv")
    df_I = pd.read_csv("Data/processed_I_fixo.csv")
    
    # Criar figura com duas tabelas lado a lado
    fig = plt.figure(figsize=(16, 8))
    
    # Título geral com e/m referência em itálico
    title_text = 'Determinação de e/m - Resultados Experimentais'
    fig.text(0.5, 0.94, title_text, ha='center', fontsize=20, fontweight='bold',
             transform=fig.transFigure)
    
    # Valor de referência (separado, com espaçamento)
    ref_text = f'e/m (referência) = {e_over_m_ref:.5e} C/kg'
    fig.text(0.5, 0.90, ref_text, ha='center', fontsize=16, style='italic')
    
    # === TABELA 1: V_fixo ===
    ax1 = fig.add_subplot(121)
    ax1.axis('tight')
    ax1.axis('off')
    
    # Título
    ax1.text(0.5, 0.95, 'Tensão Fixa (V)', ha='center', fontsize=14, 
             weight='bold', transform=ax1.transAxes)
    
    # Preparar dados da tabela V_fixo
    table_data_V = []
    cell_colors_V = []  # Para coloração condicional
    
    # Extrair ordem de grandeza comum
    e_over_m_magnitude = 1e11  # C/kg
    
    for idx, row in df_V.iterrows():
        V_fixo = row['V_fixo']
        I = row['I']
        R_cm = row['R_cm']
        e_over_m = row['e_over_m']
        u_e_over_m = row['u_e_over_m']
        diff_percent = row['diff_percent']
        
        # Converter para unidades da ordem de grandeza
        e_over_m_scaled = e_over_m / e_over_m_magnitude
        u_e_over_m_scaled = u_e_over_m / e_over_m_magnitude
        
        # Formatar
        val_str, unc_str = format_uncertainty(e_over_m_scaled, u_e_over_m_scaled)
        
        table_data_V.append([
            f'{V_fixo:.1f}',
            f'{I:.3f}',
            f'{R_cm:.0f}',
            f'({val_str} ± {unc_str})'
        ])
        
        # Determinar cor para última coluna (e/m)
        if diff_percent < 10:
            cell_colors_V.append('green')
        elif diff_percent > 100:
            cell_colors_V.append('red')
        else:
            cell_colors_V.append('black')
    
    # Cabeçalhos
    headers_V = ['V (V)', 'I (A)', 'R (cm)', 'e/m (×10¹¹ C/kg)']
    
    # Criar tabela
    table_V = ax1.table(
        cellText=table_data_V,
        colLabels=headers_V,
        cellLoc='center',
        loc='center',
        bbox=[0.05, 0.12, 0.9, 0.73]
    )
    
    # Formatação
    table_V.auto_set_font_size(False)
    table_V.set_fontsize(10)
    table_V.scale(1, 2.2)
    
    # Estilizar cabeçalho
    for i in range(4):
        table_V[(0, i)].set_facecolor('#4472C4')
        table_V[(0, i)].set_text_props(weight='bold', color='white', fontsize=11)
    
    # Alternar cores de fundo e aplicar coloração condicional
    for i in range(1, len(table_data_V) + 1):
        for j in range(4):
            # Fundo cinza mais escuro para linhas pares
            if i % 2 == 0:
                table_V[(i, j)].set_facecolor('#D0D0D0')
            
            # Coloração condicional da última coluna (e/m)
            if j == 3:
                color = cell_colors_V[i-1]
                table_V[(i, j)].set_text_props(color=color, weight='bold')
    
    # === TABELA 2: I_fixo ===
    ax2 = fig.add_subplot(122)
    ax2.axis('tight')
    ax2.axis('off')
    
    # Título (sem mencionar corrente)
    ax2.text(0.5, 0.95, 'Corrente Fixa (I)', 
             ha='center', fontsize=14, weight='bold', transform=ax2.transAxes)
    
    # Preparar dados da tabela I_fixo
    table_data_I = []
    cell_colors_I = []
    
    I_fixo_val = df_I['I_fixo'].iloc[0]
    
    for idx, row in df_I.iterrows():
        I = row['I_fixo']
        V = row['V']
        R_cm = row['R_cm']
        e_over_m = row['e_over_m']
        u_e_over_m = row['u_e_over_m']
        diff_percent = row['diff_percent']
        
        # Converter para unidades da ordem de grandeza
        e_over_m_scaled = e_over_m / e_over_m_magnitude
        u_e_over_m_scaled = u_e_over_m / e_over_m_magnitude
        
        # Formatar
        val_str, unc_str = format_uncertainty(e_over_m_scaled, u_e_over_m_scaled)
        
        table_data_I.append([
            f'{I:.3f}',
            f'{V:.1f}',
            f'{R_cm:.0f}',
            f'({val_str} ± {unc_str})'
        ])
        
        # Determinar cor para última coluna
        if diff_percent < 10:
            cell_colors_I.append('green')
        elif diff_percent > 100:
            cell_colors_I.append('red')
        else:
            cell_colors_I.append('black')
    
    # Cabeçalhos (agora com coluna I)
    headers_I = ['I (A)', 'V (V)', 'R (cm)', 'e/m (×10¹¹ C/kg)']
    
    # Criar tabela
    table_I = ax2.table(
        cellText=table_data_I,
        colLabels=headers_I,
        cellLoc='center',
        loc='center',
        bbox=[0.05, 0.12, 0.9, 0.73]
    )
    
    # Formatação
    table_I.auto_set_font_size(False)
    table_I.set_fontsize(10)
    table_I.scale(1, 2.2)
    
    # Estilizar cabeçalho
    for i in range(4):
        table_I[(0, i)].set_facecolor('#4472C4')
        table_I[(0, i)].set_text_props(weight='bold', color='white', fontsize=11)
    
    # Alternar cores de fundo e aplicar coloração condicional
    for i in range(1, len(table_data_I) + 1):
        for j in range(4):
            # Fundo cinza mais escuro para linhas pares
            if i % 2 == 0:
                table_I[(i, j)].set_facecolor('#D0D0D0')
            
            # Coloração condicional da última coluna (e/m)
            if j == 3:
                color = cell_colors_I[i-1]
                table_I[(i, j)].set_text_props(color=color, weight='bold')
    
    # Adicionar legenda com patches coloridos
    from matplotlib.patches import Rectangle
    
    # Posições para a legenda
    legend_y = 0.05
    legend_x_start = 0.25
    box_size = 0.015
    spacing = 0.17
    
    # Quadrado verde
    rect_green = Rectangle((legend_x_start, legend_y), box_size, box_size*1.5,
                           transform=fig.transFigure, facecolor='green',
                           edgecolor='black', linewidth=1)
    fig.patches.append(rect_green)
    fig.text(legend_x_start + box_size + 0.01, legend_y + box_size*0.75,
             'erro < 10%', ha='left', va='center', fontsize=11)
    
    # Quadrado vermelho
    rect_red = Rectangle((legend_x_start + spacing, legend_y), box_size, box_size*1.5,
                         transform=fig.transFigure, facecolor='red',
                         edgecolor='black', linewidth=1)
    fig.patches.append(rect_red)
    fig.text(legend_x_start + spacing + box_size + 0.01, legend_y + box_size*0.75,
             'erro > 100%', ha='left', va='center', fontsize=11)
    
    # Quadrado cinza
    rect_gray = Rectangle((legend_x_start + 2*spacing, legend_y), box_size, box_size*1.5,
                          transform=fig.transFigure, facecolor='#D0D0D0',
                          edgecolor='black', linewidth=1)
    fig.patches.append(rect_gray)
    fig.text(legend_x_start + 2*spacing + box_size + 0.01, legend_y + box_size*0.75,
             'medida "mais precisa"', ha='left', va='center', fontsize=11)
    
    # Salvar
    output_path = Path('Graficos')
    output_path.mkdir(exist_ok=True)
    output_file = output_path / 'tabela_e_over_m.png'
    
    plt.tight_layout(rect=[0, 0.08, 1, 0.88])
    fig.savefig(output_file, dpi=200, bbox_inches='tight')
    plt.close(fig)
    
    print(f"\n✅ Tabela salva em: {output_file}")

def calculate_R_theoretical(V, I):
    """
    Calcula o raio teórico R usando e/m de referência.
    De: e/m = 2V / (B*R)² onde B = k*μ₀*N*I/r
    Isolando R: R = sqrt(2*V) / (B * sqrt(e/m))
    """
    B = k * mu0 * N * I / r_bobina
    R_m = np.sqrt(2 * V) / (B * np.sqrt(e_over_m_ref))
    R_cm = R_m * 100  # metros -> cm
    return R_cm

def create_V_fixo_plot():
    """
    Cria gráfico I vs R para V fixo (duas curvas: V=61V e V=169.8V).
    """
    # Carregar dados
    df = pd.read_csv("Data/processed_V_fixo.csv")
    
    # Separar por tensão fixa
    df_V1 = df[df['V_fixo'] == 61.0]
    df_V2 = df[df['V_fixo'] == 169.8]
    
    # Criar figura
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Cores
    colors = ['#E74C3C', '#3498DB']  # Vermelho e Azul
    
    # Plotar V = 61V (dados experimentais)
    ax.errorbar(df_V1['I'], df_V1['R_cm'], 
                xerr=df_V1['u_I'], yerr=df_V1['u_R_cm'],
                fmt='o', color=colors[0], markersize=8, capsize=5, capthick=2,
                label=f'V = 61.0 V (dados)', linewidth=2, elinewidth=1.5)
    
    # Curva teórica V = 61V
    I_range = np.linspace(df_V1['I'].min()*0.8, df_V1['I'].max()*1.2, 100)
    R_theo_V1 = [calculate_R_theoretical(61.0, I) for I in I_range]
    ax.plot(I_range, R_theo_V1, '--', color=colors[0], linewidth=2.5,
            label='V = 61.0 V (teórico)', alpha=0.7)
    
    # Plotar V = 169.8V (dados experimentais)
    ax.errorbar(df_V2['I'], df_V2['R_cm'],
                xerr=df_V2['u_I'], yerr=df_V2['u_R_cm'],
                fmt='s', color=colors[1], markersize=8, capsize=5, capthick=2,
                label=f'V = 169.8 V (dados)', linewidth=2, elinewidth=1.5)
    
    # Curva teórica V = 169.8V
    I_range2 = np.linspace(df_V2['I'].min()*0.8, df_V2['I'].max()*1.2, 100)
    R_theo_V2 = [calculate_R_theoretical(169.8, I) for I in I_range2]
    ax.plot(I_range2, R_theo_V2, '--', color=colors[1], linewidth=2.5,
            label='V = 169.8 V (teórico)', alpha=0.7)
    
    # Configurações
    ax.set_xlabel('Corrente I (A)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Raio R (cm)', fontsize=13, fontweight='bold')
    ax.set_title('Tensão Fixa: I vs R\ne/m (referência) = 1.75882×10¹¹ C/kg',
                 fontsize=15, fontweight='bold', pad=15)
    
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=1)
    ax.legend(fontsize=12, framealpha=0.95, shadow=True, loc='best')
    
    # Salvar
    output_path = Path('Graficos')
    output_path.mkdir(exist_ok=True)
    output_file = output_path / 'grafico_V_fixo.png'
    
    plt.tight_layout()
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"\n✅ Gráfico V_fixo salvo em: {output_file}")

def create_I_fixo_plot():
    """
    Cria gráfico V vs R para I fixo.
    """
    # Carregar dados
    df = pd.read_csv("Data/processed_I_fixo.csv")
    
    I_fixo = df['I_fixo'].iloc[0]
    
    # Criar figura
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plotar dados experimentais
    ax.errorbar(df['V'], df['R_cm'],
                xerr=df['u_V'], yerr=df['u_R_cm'],
                fmt='o', color='#27AE60', markersize=8, capsize=5, capthick=2,
                label=f'I = {I_fixo:.3f} A (dados)', linewidth=2, elinewidth=1.5)
    
    # Curva teórica
    V_range = np.linspace(df['V'].min()*0.9, df['V'].max()*1.1, 100)
    R_theo = [calculate_R_theoretical(V, I_fixo) for V in V_range]
    ax.plot(V_range, R_theo, '--', color='#27AE60', linewidth=2.5,
            label=f'I = {I_fixo:.3f} A (teórico)', alpha=0.7)
    
    # Configurações
    ax.set_xlabel('Tensão V (V)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Raio R (cm)', fontsize=13, fontweight='bold')
    ax.set_title(f'Corrente Fixa (I = {I_fixo:.3f} A): V vs R\ne/m (referência) = 1.75882×10¹¹ C/kg',
                 fontsize=15, fontweight='bold', pad=15)
    
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=1)
    ax.legend(fontsize=12, framealpha=0.95, shadow=True, loc='best')
    
    # Salvar
    output_path = Path('Graficos')
    output_path.mkdir(exist_ok=True)
    output_file = output_path / 'grafico_I_fixo.png'
    
    plt.tight_layout()
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"\n✅ Gráfico I_fixo salvo em: {output_file}")

def print_summary():
    """
    Imprime resumo estatístico dos resultados.
    """
    df_V = pd.read_csv("Data/processed_V_fixo.csv")
    df_I = pd.read_csv("Data/processed_I_fixo.csv")
    
    print("\n" + "="*70)
    print("RESUMO ESTATÍSTICO - e/m")
    print("="*70)
    
    print("\n📊 V_fixo (Tensão Fixa):")
    print(f"   e/m médio: {df_V['e_over_m'].mean():.3e} ± {df_V['e_over_m'].std():.3e} C/kg")
    print(f"   Diferença média do ref: {df_V['diff_percent'].mean():.2f}%")
    print(f"   Faixa: {df_V['e_over_m'].min():.3e} - {df_V['e_over_m'].max():.3e} C/kg")
    
    print("\n📊 I_fixo (Corrente Fixa):")
    print(f"   e/m médio: {df_I['e_over_m'].mean():.3e} ± {df_I['e_over_m'].std():.3e} C/kg")
    print(f"   Diferença média do ref: {df_I['diff_percent'].mean():.2f}%")
    print(f"   Faixa: {df_I['e_over_m'].min():.3e} - {df_I['e_over_m'].max():.3e} C/kg")
    
    print(f"\n🎯 Valor de Referência:")
    print(f"   e/m (CODATA 2018): {e_over_m_ref:.5e} C/kg")
    
    print("\n" + "="*70)

def main():
    """
    Função principal - gera todos os gráficos e tabelas.
    """
    print("="*70)
    print("ANÁLISE DE e/m - GRÁFICOS E TABELAS")
    print("="*70)
    
    # Gerar tabela combinada
    print("\n📋 Gerando tabela combinada...")
    create_combined_table()
    
    # Gerar gráfico V_fixo
    print("\n📊 Gerando gráfico V_fixo (I vs R)...")
    create_V_fixo_plot()
    
    # Gerar gráfico I_fixo
    print("\n📊 Gerando gráfico I_fixo (V vs R)...")
    create_I_fixo_plot()
    
    # Resumo estatístico
    print_summary()
    
    print("\n✅ Análise concluída!")
    print("\n📁 Arquivos gerados:")
    print("   - Graficos/tabela_e_over_m.png")
    print("   - Graficos/grafico_V_fixo.png")
    print("   - Graficos/grafico_I_fixo.png")

if __name__ == "__main__":
    main()
