"""
test_V_minus_30.py

Cria tabela idêntica à tabela_e_over_m.png, mas com V-30 em todas as medidas.
Testa hipótese de erro sistemático de 30V.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.patches import Rectangle

# =========================
# CONSTANTES
# =========================
e_over_m_ref = 1.758820024e11  # C/kg (CODATA 2018)
N = 154
r_bobina = 0.398  # metros
u_r_bobina = 0.001
mu0 = 1.25663706212e-6
k = 0.716

# =========================
# PARÂMETRO AJUSTÁVEL
# =========================
V_OFFSET = 23  # Offset de tensão a ser subtraído (V)

def format_uncertainty(value, uncertainty):
    """Formata valor e incerteza seguindo as regras de algarismos significativos."""
    if uncertainty == 0 or np.isnan(uncertainty):
        return f"{value:.3e}", "0"
    
    if uncertainty > 0:
        magnitude = 10 ** np.floor(np.log10(uncertainty))
    else:
        magnitude = 1
    
    uncertainty_rounded = np.round(uncertainty / magnitude) * magnitude
    
    if magnitude >= 1:
        decimals = 0
    else:
        decimals = int(-np.floor(np.log10(magnitude)))
    
    value_rounded = np.round(value / magnitude) * magnitude
    
    if decimals == 0:
        value_str = f"{value_rounded:.0f}"
        uncertainty_str = f"{uncertainty_rounded:.0f}"
    else:
        value_str = f"{value_rounded:.{decimals}f}"
        uncertainty_str = f"{uncertainty_rounded:.{decimals}f}"
    
    return value_str, uncertainty_str

def calculate_e_over_m(V, I, R_cm):
    """Calcula e/m dado V, I e R."""
    R_m = R_cm / 100.0
    B = k * mu0 * N * I / r_bobina
    e_over_m = 2 * V / (B * R_m)**2
    return e_over_m

def calculate_uncertainty(V, u_V, I, u_I, R_cm, u_R_cm, e_over_m):
    """Calcula incerteza de e/m por propagação de erros."""
    R_m = R_cm / 100.0
    u_R_m = u_R_cm / 100.0
    
    term_V = (u_V / V)**2 if V != 0 else 0
    term_r = (2 * u_r_bobina / r_bobina)**2
    term_I = (2 * u_I / I)**2 if I != 0 else 0
    term_R = (2 * u_R_m / R_m)**2 if R_m != 0 else 0
    
    u_e_over_m = e_over_m * np.sqrt(term_V + term_r + term_I + term_R)
    
    return u_e_over_m

def create_table_V_minus_30():
    """
    Cria tabela idêntica à original, mas com V-30 em todas as medidas.
    """
    # Carregar dados originais
    df_V_orig = pd.read_csv("Data/processed_V_fixo.csv")
    df_I_orig = pd.read_csv("Data/processed_I_fixo.csv")
    
    print("="*70)
    print("TESTE: V - {} em todas as medidas".format(V_OFFSET))
    print("="*70)
    
    # Criar figura
    fig = plt.figure(figsize=(16, 8))
    
    # Título
    title_text = f'Determinação de e/m - Resultados com V-{V_OFFSET}'
    fig.text(0.5, 0.94, title_text, ha='center', fontsize=20, fontweight='bold',
             transform=fig.transFigure)
    
    # Valor de referência
    ref_text = f'e/m (referência) = {e_over_m_ref:.5e} C/kg'
    fig.text(0.5, 0.90, ref_text, ha='center', fontsize=16, style='italic')
    
    # === TABELA 1: V_fixo ===
    ax1 = fig.add_subplot(121)
    ax1.axis('tight')
    ax1.axis('off')
    
    ax1.text(0.5, 0.95, f'Tensão Fixa (V-{V_OFFSET})', ha='center', fontsize=14, 
             weight='bold', transform=ax1.transAxes)
    
    table_data_V = []
    cell_colors_V = []
    e_over_m_magnitude = 1e11
    
    print("\n📊 V_fixo (Tensão Fixa):")
    print("-"*70)
    
    for idx, row in df_V_orig.iterrows():
        V_orig = row['V_fixo']
        V_new = V_orig - V_OFFSET  # Subtrair offset
        I = row['I']
        R_cm = row['R_cm']
        u_I = row['u_I']
        u_R_cm = 0.5  # Incerteza fixa de 0.5 cm para o raio
        u_V = 0.1  # Incerteza da tensão
        
        # Calcular e/m com V-30
        e_over_m = calculate_e_over_m(V_new, I, R_cm)
        u_e_over_m = calculate_uncertainty(V_new, u_V, I, u_I, R_cm, u_R_cm, e_over_m)
        diff_percent = abs(e_over_m - e_over_m_ref) / e_over_m_ref * 100
        
        print(f"V={V_orig:.1f}V → V-30={V_new:.1f}V, I={I:.3f}A, R={R_cm}cm")
        print(f"  e/m = {e_over_m:.3e} C/kg (erro: {diff_percent:.1f}%)")
        
        # Escalar
        e_over_m_scaled = e_over_m / e_over_m_magnitude
        u_e_over_m_scaled = u_e_over_m / e_over_m_magnitude
        
        # Formatar
        val_str, unc_str = format_uncertainty(e_over_m_scaled, u_e_over_m_scaled)
        
        table_data_V.append([
            f'{V_new:.1f}',
            f'{I:.3f}',
            f'{R_cm:.0f}',
            f'({val_str} ± {unc_str})'
        ])
        
        # Coloração
        if diff_percent < 10:
            cell_colors_V.append('green')
        elif diff_percent > 100:
            cell_colors_V.append('red')
        else:
            cell_colors_V.append('black')
    
    # Cabeçalhos
    headers_V = [f'V-{V_OFFSET} (V)', 'I (A)', 'R (cm)', 'e/m (×10¹¹ C/kg)']
    
    # Criar tabela
    table_V = ax1.table(
        cellText=table_data_V,
        colLabels=headers_V,
        cellLoc='center',
        loc='center',
        bbox=[0.05, 0.12, 0.9, 0.73]
    )
    
    table_V.auto_set_font_size(False)
    table_V.set_fontsize(10)
    table_V.scale(1, 2.2)
    
    # Estilizar cabeçalho
    for i in range(4):
        table_V[(0, i)].set_facecolor('#4472C4')
        table_V[(0, i)].set_text_props(weight='bold', color='white', fontsize=11)
    
    # Cores e formatação
    for i in range(1, len(table_data_V) + 1):
        for j in range(4):
            if i % 2 == 0:
                table_V[(i, j)].set_facecolor('#D0D0D0')
            if j == 3:
                color = cell_colors_V[i-1]
                table_V[(i, j)].set_text_props(color=color, weight='bold')
    
    # === TABELA 2: I_fixo ===
    ax2 = fig.add_subplot(122)
    ax2.axis('tight')
    ax2.axis('off')
    
    ax2.text(0.5, 0.95, f'Corrente Fixa (I) com V-{V_OFFSET}', 
             ha='center', fontsize=14, weight='bold', transform=ax2.transAxes)
    
    table_data_I = []
    cell_colors_I = []
    
    I_fixo_val = df_I_orig['I_fixo'].iloc[0]
    
    print("\n📊 I_fixo (Corrente Fixa):")
    print("-"*70)
    
    for idx, row in df_I_orig.iterrows():
        I = row['I_fixo']
        V_orig = row['V']
        V_new = V_orig - V_OFFSET  # Subtrair offset
        R_cm = row['R_cm']
        u_I = 0.001  # Incerteza padrão para I fixo
        u_V = row['u_V']
        u_R_cm = 0.5  # Incerteza fixa de 0.5 cm para o raio
        
        # Calcular e/m com V-30
        e_over_m = calculate_e_over_m(V_new, I, R_cm)
        u_e_over_m = calculate_uncertainty(V_new, u_V, I, u_I, R_cm, u_R_cm, e_over_m)
        diff_percent = abs(e_over_m - e_over_m_ref) / e_over_m_ref * 100
        
        print(f"V={V_orig:.1f}V → V-30={V_new:.1f}V, I={I:.3f}A, R={R_cm}cm")
        print(f"  e/m = {e_over_m:.3e} C/kg (erro: {diff_percent:.1f}%)")
        
        # Escalar
        e_over_m_scaled = e_over_m / e_over_m_magnitude
        u_e_over_m_scaled = u_e_over_m / e_over_m_magnitude
        
        # Formatar
        val_str, unc_str = format_uncertainty(e_over_m_scaled, u_e_over_m_scaled)
        
        table_data_I.append([
            f'{I:.3f}',
            f'{V_new:.1f}',
            f'{R_cm:.0f}',
            f'({val_str} ± {unc_str})'
        ])
        
        # Coloração
        if diff_percent < 10:
            cell_colors_I.append('green')
        elif diff_percent > 100:
            cell_colors_I.append('red')
        else:
            cell_colors_I.append('black')
    
    # Cabeçalhos
    headers_I = ['I (A)', f'V-{V_OFFSET} (V)', 'R (cm)', 'e/m (×10¹¹ C/kg)']
    
    # Criar tabela
    table_I = ax2.table(
        cellText=table_data_I,
        colLabels=headers_I,
        cellLoc='center',
        loc='center',
        bbox=[0.05, 0.12, 0.9, 0.73]
    )
    
    table_I.auto_set_font_size(False)
    table_I.set_fontsize(10)
    table_I.scale(1, 2.2)
    
    # Estilizar cabeçalho
    for i in range(4):
        table_I[(0, i)].set_facecolor('#4472C4')
        table_I[(0, i)].set_text_props(weight='bold', color='white', fontsize=11)
    
    # Cores e formatação
    for i in range(1, len(table_data_I) + 1):
        for j in range(4):
            if i % 2 == 0:
                table_I[(i, j)].set_facecolor('#D0D0D0')
            if j == 3:
                color = cell_colors_I[i-1]
                table_I[(i, j)].set_text_props(color=color, weight='bold')
    
    # Legenda
    legend_y = 0.05
    legend_x_start = 0.25
    box_size = 0.015
    spacing = 0.17
    
    rect_green = Rectangle((legend_x_start, legend_y), box_size, box_size*1.5,
                           transform=fig.transFigure, facecolor='green',
                           edgecolor='black', linewidth=1)
    fig.patches.append(rect_green)
    fig.text(legend_x_start + box_size + 0.01, legend_y + box_size*0.75,
             'erro < 10%', ha='left', va='center', fontsize=11)
    
    rect_red = Rectangle((legend_x_start + spacing, legend_y), box_size, box_size*1.5,
                         transform=fig.transFigure, facecolor='red',
                         edgecolor='black', linewidth=1)
    fig.patches.append(rect_red)
    fig.text(legend_x_start + spacing + box_size + 0.01, legend_y + box_size*0.75,
             'erro > 100%', ha='left', va='center', fontsize=11)
    
    rect_gray = Rectangle((legend_x_start + 2*spacing, legend_y), box_size, box_size*1.5,
                          transform=fig.transFigure, facecolor='#D0D0D0',
                          edgecolor='black', linewidth=1)
    fig.patches.append(rect_gray)
    fig.text(legend_x_start + 2*spacing + box_size + 0.01, legend_y + box_size*0.75,
             'medida "mais precisa"', ha='left', va='center', fontsize=11)
    
    # Salvar
    output_path = Path('Graficos')
    output_path.mkdir(exist_ok=True)
    output_file = output_path / 'tabela_e_over_m_V_minus_30.png'
    
    plt.tight_layout(rect=[0, 0.08, 1, 0.88])
    fig.savefig(output_file, dpi=200, bbox_inches='tight')
    plt.close(fig)
    
    print("\n" + "="*70)
    print(f"✅ Tabela salva em: {output_file}")
    print("="*70)

if __name__ == "__main__":
    create_table_V_minus_30()
