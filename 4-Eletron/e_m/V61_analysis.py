"""
V61_analysis.py

Análise específica dos dados de V=61V para encontrar o valor real de tensão
através de fit dos dados experimentais.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import curve_fit
from matplotlib.patches import Rectangle

# =========================
# CONSTANTES
# =========================
e_over_m_ref = 1.758820024e11  # C/kg (CODATA 2018)
N = 154
r_bobina = 0.398  # metros
mu0 = 1.25663706212e-6
k = 0.716

def format_uncertainty(value, uncertainty):
    """
    Formata valor e incerteza seguindo as regras de algarismos significativos.
    """
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

def calculate_R_theoretical(V, I):
    """
    Calcula o raio teórico R dado V e I.
    """
    B = k * mu0 * N * I / r_bobina
    R_m = np.sqrt(2 * V) / (B * np.sqrt(e_over_m_ref))
    R_cm = R_m * 100  # metros -> cm
    return R_cm

def R_fit_function(I, V_fit):
    """
    Função para fit: R em função de I, com V como parâmetro ajustável.
    """
    return calculate_R_theoretical(V_fit, I)

def calculate_e_over_m(V, I, R_cm):
    """
    Calcula e/m dado V, I e R.
    """
    R_m = R_cm / 100.0
    B = k * mu0 * N * I / r_bobina
    e_over_m = 2 * V / (B * R_m)**2
    return e_over_m

def calculate_uncertainty(V, u_V, I, u_I, R_cm, u_R_cm, e_over_m):
    """
    Calcula incerteza de e/m por propagação de erros.
    """
    R_m = R_cm / 100.0
    u_R_m = u_R_cm / 100.0
    
    u_r_bobina = 0.001
    
    term_V = (u_V / V)**2 if V != 0 else 0
    term_r = (2 * u_r_bobina / r_bobina)**2
    term_I = (2 * u_I / I)**2 if I != 0 else 0
    term_R = (2 * u_R_m / R_m)**2 if R_m != 0 else 0
    
    u_e_over_m = e_over_m * np.sqrt(term_V + term_r + term_I + term_R)
    
    return u_e_over_m

def create_V61_fit_plot():
    """
    Cria gráfico com fit para encontrar V real dos dados de 61V.
    """
    # Carregar dados
    df = pd.read_csv("Data/processed_V_fixo.csv")
    df_V61 = df[df['V_fixo'] == 61.0].copy()
    
    print("="*70)
    print("ANÁLISE DE FIT - V = 61V")
    print("="*70)
    
    print(f"\n📊 Dados de V=61V: {len(df_V61)} pontos")
    
    # Extrair dados
    I_data = df_V61['I'].values
    R_data = df_V61['R_cm'].values
    u_I = df_V61['u_I'].values
    u_R = df_V61['u_R_cm'].values
    
    # Realizar fit para encontrar V ótimo
    print("\n🔍 Realizando fit para encontrar V...")
    
    # Parâmetros iniciais: V_inicial = 61V
    p0 = [61.0]
    
    # Fit
    popt, pcov = curve_fit(R_fit_function, I_data, R_data, p0=p0, sigma=u_R, absolute_sigma=True)
    V_fitted = popt[0]
    V_fitted_err = np.sqrt(pcov[0, 0])
    
    print(f"\n✅ Resultado do fit:")
    print(f"   V (nominal) = 61.0 V")
    print(f"   V (estimado) = {V_fitted:.2f} ± {V_fitted_err:.2f} V")
    print(f"   Diferença: {abs(V_fitted - 61.0):.2f} V ({abs(V_fitted - 61.0)/61.0 * 100:.1f}%)")
    
    # Criar gráfico
    fig, ax = plt.subplots(figsize=(12, 8))
    
    color = '#E74C3C'  # Vermelho
    
    # Dados experimentais
    ax.errorbar(I_data, R_data,
                xerr=u_I, yerr=u_R,
                fmt='o', color=color, markersize=10, capsize=5, capthick=2,
                label='Dados experimentais', linewidth=2, elinewidth=1.5, zorder=3)
    
    # Curva teórica (V=61V nominal)
    I_range = np.linspace(I_data.min()*0.8, I_data.max()*1.2, 100)
    R_theo_nominal = [calculate_R_theoretical(61.0, I) for I in I_range]
    ax.plot(I_range, R_theo_nominal, '--', color=color, linewidth=2.5,
            label='V = 61.0 V (teórico)', alpha=0.7, zorder=2)
    
    # Curva ajustada (V fitted)
    R_theo_fitted = [calculate_R_theoretical(V_fitted, I) for I in I_range]
    ax.plot(I_range, R_theo_fitted, '-.', color='green', linewidth=3,
            label=f'V = {V_fitted:.2f} V (estimado)', alpha=0.8, zorder=2)
    
    # Adicionar texto com V estimado
    text_box = f'V (estimado) = {V_fitted:.2f} ± {V_fitted_err:.2f} V'
    ax.text(0.05, 0.95, text_box, transform=ax.transAxes,
            fontsize=13, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9, edgecolor='black', linewidth=2))
    
    # Configurações
    ax.set_xlabel('Corrente I (A)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Raio R (cm)', fontsize=13, fontweight='bold')
    ax.set_title('Análise de Fit - Estimativa de Tensão Real\nV nominal = 61.0 V',
                 fontsize=15, fontweight='bold', pad=15)
    
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=1)
    ax.legend(fontsize=12, framealpha=0.95, shadow=True, loc='best')
    
    # Salvar
    output_path = Path('Graficos')
    output_path.mkdir(exist_ok=True)
    output_file = output_path / 'V61_fit_analysis.png'
    
    plt.tight_layout()
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"\n✅ Gráfico salvo em: {output_file}")
    
    return V_fitted, V_fitted_err, df_V61

def create_V61_comparison_table(V_fitted, V_fitted_err, df_V61):
    """
    Cria tabela comparando e/m calculado com V nominal vs V ajustado.
    """
    print("\n📋 Gerando tabela comparativa...")
    
    # Preparar dados da tabela
    table_data = []
    cell_colors = []
    
    e_over_m_magnitude = 1e11
    
    for idx, row in df_V61.iterrows():
        I = row['I']
        R_cm = row['R_cm']
        u_I = row['u_I']
        u_R_cm = row['u_R_cm']
        
        # e/m antigo (V=61V)
        e_over_m_old = row['e_over_m']
        u_e_over_m_old = row['u_e_over_m']
        diff_old = row['diff_percent']
        
        # e/m novo (V ajustado)
        e_over_m_new = calculate_e_over_m(V_fitted, I, R_cm)
        u_e_over_m_new = calculate_uncertainty(V_fitted, V_fitted_err, I, u_I, R_cm, u_R_cm, e_over_m_new)
        diff_new = abs(e_over_m_new - e_over_m_ref) / e_over_m_ref * 100
        
        # Escalar
        e_over_m_old_scaled = e_over_m_old / e_over_m_magnitude
        u_e_over_m_old_scaled = u_e_over_m_old / e_over_m_magnitude
        e_over_m_new_scaled = e_over_m_new / e_over_m_magnitude
        u_e_over_m_new_scaled = u_e_over_m_new / e_over_m_magnitude
        
        # Formatar
        val_old_str, unc_old_str = format_uncertainty(e_over_m_old_scaled, u_e_over_m_old_scaled)
        val_new_str, unc_new_str = format_uncertainty(e_over_m_new_scaled, u_e_over_m_new_scaled)
        
        table_data.append([
            f'{I:.3f}',
            f'{R_cm:.0f}',
            f'({val_old_str} ± {unc_old_str})',
            f'({val_new_str} ± {unc_new_str})'
        ])
        
        # Coloração para e/m novo
        if diff_new < 10:
            cell_colors.append('green')
        elif diff_new > 100:
            cell_colors.append('red')
        else:
            cell_colors.append('black')
    
    # Criar figura
    fig = plt.figure(figsize=(14, 8))
    
    # Título
    fig.suptitle('Comparação: e/m com V Nominal vs V Ajustado', 
                 fontsize=18, fontweight='bold', y=0.94)
    
    # Subtítulo com valores
    subtitle = f'V (nominal) = 61.0 V  |  V (estimado) = {V_fitted:.2f} ± {V_fitted_err:.2f} V'
    fig.text(0.5, 0.88, subtitle, ha='center', fontsize=14, style='italic')
    
    # Valor de referência
    ref_text = f'e/m (referência) = {e_over_m_ref:.5e} C/kg'
    fig.text(0.5, 0.84, ref_text, ha='center', fontsize=13, style='italic')
    
    # Criar eixo
    ax = fig.add_subplot(111)
    ax.axis('tight')
    ax.axis('off')
    
    # Cabeçalhos
    headers = ['I (A)', 'R (cm)', 'e/m antigo (×10¹¹ C/kg)', 'e/m novo (×10¹¹ C/kg)']
    
    # Criar tabela
    table = ax.table(
        cellText=table_data,
        colLabels=headers,
        cellLoc='center',
        loc='center',
        bbox=[0.1, 0.25, 0.8, 0.5]
    )
    
    # Formatação
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    
    # Estilizar cabeçalho
    for i in range(4):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white', fontsize=12)
    
    # Alternar cores e aplicar coloração condicional
    for i in range(1, len(table_data) + 1):
        for j in range(4):
            # Fundo cinza para linhas pares
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#D0D0D0')
            
            # Coloração condicional da última coluna (e/m novo)
            if j == 3:
                color = cell_colors[i-1]
                table[(i, j)].set_text_props(color=color, weight='bold')
    
    # Adicionar legenda com patches coloridos
    legend_y = 0.12
    legend_x_start = 0.25
    box_size = 0.02
    spacing = 0.17
    
    # Quadrado verde
    rect_green = Rectangle((legend_x_start, legend_y), box_size, box_size*1.2,
                           transform=fig.transFigure, facecolor='green',
                           edgecolor='black', linewidth=1)
    fig.patches.append(rect_green)
    fig.text(legend_x_start + box_size + 0.01, legend_y + box_size*0.6,
             'erro < 10%', ha='left', va='center', fontsize=11)
    
    # Quadrado vermelho
    rect_red = Rectangle((legend_x_start + spacing, legend_y), box_size, box_size*1.2,
                         transform=fig.transFigure, facecolor='red',
                         edgecolor='black', linewidth=1)
    fig.patches.append(rect_red)
    fig.text(legend_x_start + spacing + box_size + 0.01, legend_y + box_size*0.6,
             'erro > 100%', ha='left', va='center', fontsize=11)
    
    # Quadrado cinza
    rect_gray = Rectangle((legend_x_start + 2*spacing, legend_y), box_size, box_size*1.2,
                          transform=fig.transFigure, facecolor='#D0D0D0',
                          edgecolor='black', linewidth=1)
    fig.patches.append(rect_gray)
    fig.text(legend_x_start + 2*spacing + box_size + 0.01, legend_y + box_size*0.6,
             'medida "mais precisa"', ha='left', va='center', fontsize=11)
    
    # Salvar
    output_path = Path('Graficos')
    output_path.mkdir(exist_ok=True)
    output_file = output_path / 'V61_comparison_table.png'
    
    plt.tight_layout(rect=[0, 0.08, 1, 0.82])
    fig.savefig(output_file, dpi=200, bbox_inches='tight')
    plt.close(fig)
    
    print(f"✅ Tabela salva em: {output_file}")
    
    # Imprimir resumo
    print("\n" + "="*70)
    print("RESUMO DA COMPARAÇÃO")
    print("="*70)
    
    for idx, row in df_V61.iterrows():
        I = row['I']
        R_cm = row['R_cm']
        e_over_m_old = row['e_over_m']
        diff_old = row['diff_percent']
        
        e_over_m_new = calculate_e_over_m(V_fitted, I, R_cm)
        diff_new = abs(e_over_m_new - e_over_m_ref) / e_over_m_ref * 100
        
        print(f"\nI = {I:.3f} A, R = {R_cm:.0f} cm:")
        print(f"  e/m antigo: {e_over_m_old:.3e} C/kg (erro: {diff_old:.1f}%)")
        print(f"  e/m novo:   {e_over_m_new:.3e} C/kg (erro: {diff_new:.1f}%)")
        print(f"  Melhoria: {diff_old - diff_new:.1f} pontos percentuais")

def main():
    """
    Função principal - análise completa dos dados de V=61V.
    """
    print("\n🔬 ANÁLISE ESPECÍFICA - V = 61V")
    
    # Criar gráfico com fit
    V_fitted, V_fitted_err, df_V61 = create_V61_fit_plot()
    
    # Criar tabela comparativa
    create_V61_comparison_table(V_fitted, V_fitted_err, df_V61)
    
    print("\n✅ Análise concluída!")
    print("\n📁 Arquivos gerados:")
    print("  - Graficos/V61_fit_analysis.png")
    print("  - Graficos/V61_comparison_table.png")
    print("\n" + "="*70)

if __name__ == "__main__":
    main()
