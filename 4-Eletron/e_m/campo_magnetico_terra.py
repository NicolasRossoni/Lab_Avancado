"""
campo_magnetico_terra.py

Análise do campo magnético da Terra usando medidas paralelas e antiparalelas.
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
mu0 = 1.25663706212e-6
k = 0.716
I_fixo = 1.496  # A (corrente fixa)
B_Terra_ref = 2.3e-5  # T (≈ 0.023 mT) - valor de referência campo magnético terrestre

def parse_csv_value(value):
    """Converte valores com vírgula para float."""
    if isinstance(value, str):
        return float(value.replace(',', '.'))
    return float(value)

def calculate_e_over_m(V, I, R_cm):
    """Calcula e/m dado V, I e R."""
    R_m = R_cm / 100.0
    B = k * mu0 * N * I / r_bobina
    e_over_m = 2 * V / (B * R_m)**2
    return e_over_m

def calculate_B_from_em(V, R_cm, e_over_m_ref):
    """
    Calcula B usando a equação: B = sqrt(2*V / (e_over_m_ref * R^2))
    
    Entradas:
        V: tensão [V]
        R_cm: raio [cm]
        e_over_m_ref: razão carga-massa de referência [C/kg]
    
    Saída:
        B: campo magnético [T]
    """
    R_m = R_cm / 100.0  # converter para metros
    B = np.sqrt(2.0 * V / (e_over_m_ref * R_m**2))
    return B

def calculate_u_B(B, V, u_V, e_over_m, u_e_over_m, R, u_R):
    """
    Calcula incerteza de B por propagação.
    u_B = B * sqrt( (0.5*u_V/V)^2 + (0.5*u_e_over_m/e_over_m)^2 + (u_R/R)^2 )
    """
    term_V = (0.5 * u_V / V)**2 if V != 0 else 0
    term_em = (0.5 * u_e_over_m / e_over_m)**2 if e_over_m != 0 else 0
    term_R = (u_R / R)**2 if R != 0 else 0
    
    u_B = B * np.sqrt(term_V + term_em + term_R)
    return u_B

def calculate_B_Terra(B_par, B_aperp):
    """Calcula campo magnético da Terra: B_Terra = (B_par - B_aperp) / 2"""
    return (B_par - B_aperp) / 2.0

def calculate_u_B_Terra(u_B_par, u_B_aperp, rho=0):
    """
    Calcula incerteza do campo magnético da Terra.
    u_B_Terra = 0.5 * sqrt(u_B_par^2 + u_B_aperp^2 - 2*rho*u_B_par*u_B_aperp)
    """
    return 0.5 * np.sqrt(u_B_par**2 + u_B_aperp**2 - 2*rho*u_B_par*u_B_aperp)

def format_uncertainty(value, uncertainty):
    """Formata valor e incerteza."""
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

def create_campo_terra_tables():
    """
    Cria imagem com 3 tabelas: B_paralelo, B_antiparalelo, B_Terra.
    """
    # Carregar dados
    df_par = pd.read_csv("Data/B_paralelo.csv")
    df_aperp = pd.read_csv("Data/B_antiparalelo.csv")
    
    print("="*70)
    print("ANÁLISE DO CAMPO MAGNÉTICO DA TERRA")
    print("="*70)
    print(f"\nI (fixo) = {I_fixo} A")
    print(f"Medidas paralelas: {len(df_par)}")
    print(f"Medidas antiparalelas: {len(df_aperp)}")
    
    # Incertezas
    u_V = 0.1  # V
    u_R_cm = 0.5  # cm
    u_I = 0.001  # A
    
    # Processar dados paralelos
    data_par = []
    for idx, row in df_par.iterrows():
        V = parse_csv_value(row['Tensão (V)'])
        R_cm = float(row['R (cm)'])
        R_m = R_cm / 100.0
        u_R_m = u_R_cm / 100.0
        
        # Calcular e/m com esta medida
        e_over_m = calculate_e_over_m(V, I_fixo, R_cm)
        
        # Calcular B usando e/m de REFERÊNCIA
        B = calculate_B_from_em(V, R_cm, e_over_m_ref)
        
        # Calcular incerteza de B
        u_e_over_m_ref = 0.0  # Assumindo e/m ref sem incerteza
        u_B = calculate_u_B(B, V, u_V, e_over_m_ref, u_e_over_m_ref, R_m, u_R_m)
        
        data_par.append({
            'V': V,
            'R_cm': R_cm,
            'e_over_m': e_over_m,
            'B': B,
            'u_B': u_B
        })
    
    # Processar dados antiparalelos
    data_aperp = []
    for idx, row in df_aperp.iterrows():
        V = parse_csv_value(row['Tensão (V)'])
        R_cm = float(row['R (cm)'])
        R_m = R_cm / 100.0
        u_R_m = u_R_cm / 100.0
        
        # Calcular e/m com esta medida
        e_over_m = calculate_e_over_m(V, I_fixo, R_cm)
        
        # Calcular B usando e/m de REFERÊNCIA
        B = calculate_B_from_em(V, R_cm, e_over_m_ref)
        
        # Calcular incerteza de B
        u_e_over_m_ref = 0.0
        u_B = calculate_u_B(B, V, u_V, e_over_m_ref, u_e_over_m_ref, R_m, u_R_m)
        
        data_aperp.append({
            'V': V,
            'R_cm': R_cm,
            'e_over_m': e_over_m,
            'B': B,
            'u_B': u_B
        })
    
    # Calcular B_Terra para cada par de raio
    data_terra = []
    for i in range(len(data_par)):
        R_cm = data_par[i]['R_cm']
        B_par = data_par[i]['B']
        B_aperp = data_aperp[i]['B']
        u_B_par = data_par[i]['u_B']
        u_B_aperp = data_aperp[i]['u_B']
        
        B_Terra = calculate_B_Terra(B_par, B_aperp)
        u_B_Terra = calculate_u_B_Terra(u_B_par, u_B_aperp)
        
        data_terra.append({
            'R_cm': R_cm,
            'B_Terra': B_Terra,
            'u_B_Terra': u_B_Terra
        })
        
        print(f"\nR = {R_cm} cm:")
        print(f"  B_paralelo = {B_par*1e3:.6f} ± {u_B_par*1e3:.6f} mT")
        print(f"  B_antiparalelo = {B_aperp*1e3:.6f} ± {u_B_aperp*1e3:.6f} mT")
        print(f"  B_Terra = {B_Terra*1e6:.3f} ± {u_B_Terra*1e6:.3f} µT")
    
    # Criar figura com 3 tabelas
    fig = plt.figure(figsize=(18, 10))
    
    # Título geral
    fig.suptitle('Determinação do Campo Magnético da Terra', 
                 fontsize=20, fontweight='bold', y=0.96)
    
    # Magnitude comum
    e_over_m_magnitude = 1e11
    B_magnitude = 1e-3  # mT
    B_Terra_magnitude = 1e-6  # µT
    
    # === TABELA 1: B_paralelo ===
    ax1 = fig.add_subplot(131)
    ax1.axis('tight')
    ax1.axis('off')
    
    ax1.text(0.5, 0.95, 'Campo Paralelo', ha='center', fontsize=14, 
             weight='bold', transform=ax1.transAxes)
    
    table_data_par = []
    for d in data_par:
        B_scaled = d['B'] / B_magnitude
        u_B_scaled = d['u_B'] / B_magnitude
        
        B_val_str, B_unc_str = format_uncertainty(B_scaled, u_B_scaled)
        
        table_data_par.append([
            f"{I_fixo:.3f}",
            f"{d['V']:.1f}",
            f"{d['R_cm']:.0f}",
            f"({B_val_str} ± {B_unc_str})"
        ])
    
    headers_par = ['I (A)', 'V (V)', 'R (cm)', 'B (mT)']
    
    table_par = ax1.table(
        cellText=table_data_par,
        colLabels=headers_par,
        cellLoc='center',
        loc='center',
        bbox=[0.05, 0.2, 0.9, 0.65]
    )
    
    table_par.auto_set_font_size(False)
    table_par.set_fontsize(10)
    table_par.scale(1, 2.5)
    
    # Estilizar cabeçalho
    for i in range(4):
        table_par[(0, i)].set_facecolor('#4472C4')
        table_par[(0, i)].set_text_props(weight='bold', color='white', fontsize=11)
    
    # Alternar cores
    for i in range(1, len(table_data_par) + 1):
        for j in range(4):
            if i % 2 == 0:
                table_par[(i, j)].set_facecolor('#D0D0D0')
    
    # === TABELA 2: B_antiparalelo ===
    ax2 = fig.add_subplot(132)
    ax2.axis('tight')
    ax2.axis('off')
    
    ax2.text(0.5, 0.95, 'Campo Antiparalelo', ha='center', fontsize=14, 
             weight='bold', transform=ax2.transAxes)
    
    table_data_aperp = []
    for d in data_aperp:
        B_scaled = d['B'] / B_magnitude
        u_B_scaled = d['u_B'] / B_magnitude
        
        B_val_str, B_unc_str = format_uncertainty(B_scaled, u_B_scaled)
        
        table_data_aperp.append([
            f"{I_fixo:.3f}",
            f"{d['V']:.1f}",
            f"{d['R_cm']:.0f}",
            f"({B_val_str} ± {B_unc_str})"
        ])
    
    headers_aperp = ['I (A)', 'V (V)', 'R (cm)', 'B (mT)']
    
    table_aperp = ax2.table(
        cellText=table_data_aperp,
        colLabels=headers_aperp,
        cellLoc='center',
        loc='center',
        bbox=[0.05, 0.2, 0.9, 0.65]
    )
    
    table_aperp.auto_set_font_size(False)
    table_aperp.set_fontsize(10)
    table_aperp.scale(1, 2.5)
    
    # Estilizar cabeçalho
    for i in range(4):
        table_aperp[(0, i)].set_facecolor('#4472C4')
        table_aperp[(0, i)].set_text_props(weight='bold', color='white', fontsize=11)
    
    # Alternar cores
    for i in range(1, len(table_data_aperp) + 1):
        for j in range(4):
            if i % 2 == 0:
                table_aperp[(i, j)].set_facecolor('#D0D0D0')
    
    # === TABELA 3: B_Terra ===
    ax3 = fig.add_subplot(133)
    ax3.axis('tight')
    ax3.axis('off')
    
    ax3.text(0.5, 0.95, 'Campo Magnético da Terra', ha='center', fontsize=14, 
             weight='bold', transform=ax3.transAxes)
    
    table_data_terra = []
    for d in data_terra:
        B_Terra_scaled = d['B_Terra'] / B_Terra_magnitude
        u_B_Terra_scaled = d['u_B_Terra'] / B_Terra_magnitude
        
        # Para B_Terra: arredondar valor para inteiro, incerteza para 1 alg. significativo
        B_val_str = f"{B_Terra_scaled:.0f}"
        
        # Arredondar incerteza para 1 algarismo significativo
        if u_B_Terra_scaled > 0:
            magnitude = 10 ** np.floor(np.log10(u_B_Terra_scaled))
            u_rounded = np.round(u_B_Terra_scaled / magnitude) * magnitude
            u_unc_str = f"{u_rounded:.0f}" if magnitude >= 1 else f"{u_rounded:.1f}"
        else:
            u_unc_str = "0"
        
        table_data_terra.append([
            f"{d['R_cm']:.0f}",
            f"({B_val_str} ± {u_unc_str})"
        ])
    
    headers_terra = ['R (cm)', 'B_Terra (µT)']
    
    table_terra = ax3.table(
        cellText=table_data_terra,
        colLabels=headers_terra,
        cellLoc='center',
        loc='center',
        bbox=[0.2, 0.35, 0.6, 0.5]
    )
    
    table_terra.auto_set_font_size(False)
    table_terra.set_fontsize(11)
    table_terra.scale(1, 2.5)
    
    # Estilizar cabeçalho
    for i in range(2):
        table_terra[(0, i)].set_facecolor('#4472C4')
        table_terra[(0, i)].set_text_props(weight='bold', color='white', fontsize=12)
    
    # Alternar cores
    for i in range(1, len(table_data_terra) + 1):
        for j in range(2):
            if i % 2 == 0:
                table_terra[(i, j)].set_facecolor('#D0D0D0')
    
    # Adicionar referência do B_Terra embaixo da tabela 3
    ref_text = f'B_Terra (referência) ≈ {B_Terra_ref*1e6:.1f} µT'
    ax3.text(0.5, 0.15, ref_text, ha='center', fontsize=12, 
             style='italic', transform=ax3.transAxes)
    
    # Salvar
    output_path = Path('Graficos')
    output_path.mkdir(exist_ok=True)
    output_file = output_path / 'campo_magnetico_terra.png'
    
    plt.tight_layout(rect=[0, 0.02, 1, 0.94])
    fig.savefig(output_file, dpi=200, bbox_inches='tight')
    plt.close(fig)
    
    print(f"\n✅ Tabela salva em: {output_file}")
    print("="*70)

if __name__ == "__main__":
    create_campo_terra_tables()
