"""
d_analysis.py

Determinação dos espaçamentos cristalinos d₁ e d₂ usando difração de elétrons.

Relação: R = (h*λ) / (√(2me) * d) * V^(-1/2)
Portanto: coef_angular = (h*λ) / (√(2me) * d)
Isolando: d = (h*λ) / (√(2me) * coef_angular)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

# =========================
# CONSTANTES FÍSICAS E PARÂMETROS
# =========================
h_planck = 6.62607015e-34  # J·s (valor de referência)
m_electron = 9.11e-31      # kg
e_charge = 1.6e-19         # C
sqrt_2me = np.sqrt(2 * m_electron * e_charge)

# Distância tela-amostra (corrigida)
L = 0.135 / 3  # metros = 45 mm

# Valores de referência dos espaçamentos cristalinos (Angstrom)
d1_ref = 2.13  # Å
d2_ref = 1.23  # Å

def linear_fit(x, y):
    """
    Ajuste linear por mínimos quadrados: y = a*x + b
    
    Returns:
        slope, intercept, slope_uncertainty, intercept_uncertainty, r_squared
    """
    result = stats.linregress(x, y)
    
    slope = result.slope
    intercept = result.intercept
    slope_std = result.stderr
    intercept_std = result.intercept_stderr
    r_squared = result.rvalue ** 2
    
    return slope, intercept, slope_std, intercept_std, r_squared

def format_uncertainty(value, uncertainty):
    """
    Formata valor e incerteza seguindo as regras de algarismos significativos.
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

def create_d_plot(df_paq, df_foto, radius_column, d_ref, d_name, output_filename):
    """
    Cria gráfico de R vs V^(-1/2) para determinar espaçamento d.
    
    Args:
        df_paq: DataFrame do paquímetro
        df_foto: DataFrame da foto
        radius_column: nome da coluna do raio ('r' ou 'R')
        d_ref: valor de referência de d (Angstrom)
        d_name: nome do espaçamento ('d₁' ou 'd₂')
        output_filename: nome do arquivo de saída
    
    Returns:
        dict com resultados
    """
    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(111)
    
    # Cores
    colors = {
        'paq': '#E74C3C',   # Vermelho
        'foto': '#3498DB'   # Azul
    }
    
    # === PAQUÍMETRO ===
    V_paq = df_paq['Volts'].values
    R_paq = df_paq[radius_column].values * 1e10  # metros -> Angstrom
    V_inv_sqrt_paq = V_paq ** (-0.5)
    
    # Ajuste linear paquímetro
    slope_paq, intercept_paq, slope_std_paq, _, r2_paq = linear_fit(V_inv_sqrt_paq, R_paq)
    
    # Calcular d do paquímetro usando: d = (L * h) / (√(2me) * coef_ang)
    # R = L * λ / d = L * (h/√(2meV)) / d = (L*h)/(d*√(2me)) * V^(-1/2)
    # Portanto: slope = (L*h) / (d*√(2me))
    # Isolando: d = (L*h) / (√(2me)*slope)
    # slope está em [Å * V^(1/2)], converter para [m * V^(1/2)]
    slope_paq_m = slope_paq * 1e-10  # Å -> m
    
    d_paq_m = (L * h_planck) / (sqrt_2me * slope_paq_m)  # metros
    d_paq = d_paq_m * 1e10  # Angstrom
    
    # Propagação de incerteza: σ_d = d * (σ_slope / slope)
    d_paq_unc = d_paq * (slope_std_paq / slope_paq)
    
    # === FOTO ===
    V_foto = df_foto['Volts'].values
    R_foto = df_foto[radius_column].values * 1e10  # metros -> Angstrom
    V_inv_sqrt_foto = V_foto ** (-0.5)
    
    # Ajuste linear foto
    slope_foto, intercept_foto, slope_std_foto, _, r2_foto = linear_fit(V_inv_sqrt_foto, R_foto)
    
    # Calcular d da foto
    slope_foto_m = slope_foto * 1e-10  # Å -> m
    
    d_foto_m = (L * h_planck) / (sqrt_2me * slope_foto_m)  # metros
    d_foto = d_foto_m * 1e10  # Angstrom
    
    # Propagação de incerteza
    d_foto_unc = d_foto * (slope_std_foto / slope_foto)
    
    # === PLOTAR DADOS E AJUSTES ===
    
    # Gerar pontos para retas ajustadas
    V_inv_sqrt_min = min(V_inv_sqrt_paq.min(), V_inv_sqrt_foto.min())
    V_inv_sqrt_max = max(V_inv_sqrt_paq.max(), V_inv_sqrt_foto.max())
    V_inv_sqrt_fit = np.linspace(V_inv_sqrt_min * 0.95, V_inv_sqrt_max * 1.05, 100)
    
    R_paq_fit = slope_paq * V_inv_sqrt_fit + intercept_paq
    R_foto_fit = slope_foto * V_inv_sqrt_fit + intercept_foto
    
    # Paquímetro
    ax.scatter(V_inv_sqrt_paq, R_paq, color=colors['paq'], s=120,
               marker='s', label='Paquímetro (dados)', zorder=3, 
               edgecolors='black', linewidth=1.5)
    ax.plot(V_inv_sqrt_fit, R_paq_fit, color=colors['paq'],
            linestyle='--', linewidth=2.5, label='MMQ Paquímetro', zorder=2)
    
    # Foto
    ax.scatter(V_inv_sqrt_foto, R_foto, color=colors['foto'], s=80,
               marker='o', label='Foto (dados)', zorder=3,
               edgecolors='black', linewidth=1.5)
    ax.plot(V_inv_sqrt_fit, R_foto_fit, color=colors['foto'],
            linestyle='--', linewidth=2, label='MMQ Foto', zorder=2)
    
    # Configurações do gráfico
    ax.set_xlabel('V⁻¹/² (V⁻¹/²)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Raio R (Å)', fontsize=14, fontweight='bold')
    ax.set_title(f'Determinação do Espaçamento Cristalino {d_name}',
                 fontsize=16, fontweight='bold', pad=20)
    
    # Grade
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=1)
    
    # Legenda
    ax.legend(loc='best', fontsize=11, framealpha=0.95, shadow=True)
    
    # Adicionar informações dos ajustes (fora do gráfico)
    info_text = (
        f'Coeficientes Angulares:\n'
        f'Paquímetro: a = {slope_paq:.2f} ± {slope_std_paq:.2f} Å·V¹/²\n'
        f'Foto:       a = {slope_foto:.2f} ± {slope_std_foto:.2f} Å·V¹/²\n'
        f'\n'
        f'Espaçamento Cristalino {d_name}:\n'
        f'd_paq  = ({d_paq:.3f} ± {d_paq_unc:.3f}) Å\n'
        f'd_foto = ({d_foto:.3f} ± {d_foto_unc:.3f}) Å\n'
        f'\n'
        f'Valor de Referência:\n'
        f'{d_name}_ref = {d_ref:.2f} Å\n'
        f'\n'
        f'L = {L*1000:.1f} mm\n'
        f'h = {h_planck:.5e} J·s\n'
        f'm = {m_electron:.2e} kg\n'
        f'e = {e_charge:.2e} C'
    )
    
    # Colocar texto fora do gráfico (à direita)
    fig.text(0.72, 0.5, info_text, fontsize=9, verticalalignment='center',
             family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))
    
    # Salvar
    output_path = Path('Graficos')
    output_path.mkdir(exist_ok=True)
    output_file = output_path / f'{output_filename}.png'
    
    # Ajustar layout
    plt.subplots_adjust(right=0.7)
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"\n✅ Gráfico salvo em: {output_file}")
    
    # Retornar resultados
    results = {
        'd_name': d_name,
        'd_ref': d_ref,
        'paquimetro': {
            'slope': slope_paq,
            'slope_std': slope_std_paq,
            'd': d_paq,
            'd_unc': d_paq_unc,
            'r_squared': r2_paq
        },
        'foto': {
            'slope': slope_foto,
            'slope_std': slope_std_foto,
            'd': d_foto,
            'd_unc': d_foto_unc,
            'r_squared': r2_foto
        }
    }
    
    return results

def create_d_table(results_d1, results_d2):
    """
    Cria tabela com os valores de d determinados.
    """
    # Preparar dados da tabela
    table_data = []
    
    # d₁ Paquímetro
    val_str, unc_str = format_uncertainty(results_d1['paquimetro']['d'], 
                                          results_d1['paquimetro']['d_unc'])
    table_data.append(['d₁ Paquímetro', f'({val_str} ± {unc_str}) Å', f'{results_d1["d_ref"]:.2f} Å'])
    
    # d₁ Foto
    val_str, unc_str = format_uncertainty(results_d1['foto']['d'], 
                                          results_d1['foto']['d_unc'])
    table_data.append(['d₁ Foto', f'({val_str} ± {unc_str}) Å', f'{results_d1["d_ref"]:.2f} Å'])
    
    # d₂ Paquímetro
    val_str, unc_str = format_uncertainty(results_d2['paquimetro']['d'], 
                                          results_d2['paquimetro']['d_unc'])
    table_data.append(['d₂ Paquímetro', f'({val_str} ± {unc_str}) Å', f'{results_d2["d_ref"]:.2f} Å'])
    
    # d₂ Foto
    val_str, unc_str = format_uncertainty(results_d2['foto']['d'], 
                                          results_d2['foto']['d_unc'])
    table_data.append(['d₂ Foto', f'({val_str} ± {unc_str}) Å', f'{results_d2["d_ref"]:.2f} Å'])
    
    # Criar figura
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis('tight')
    ax.axis('off')
    
    # Título
    title_text = 'Determinação dos Espaçamentos Cristalinos'
    ax.text(0.5, 0.95, title_text, ha='center', fontsize=16, weight='bold',
            transform=ax.transAxes)
    
    # Cabeçalhos
    headers = ['Método', 'Valor Medido', 'Valor de Referência']
    
    # Criar tabela
    table = ax.table(
        cellText=table_data,
        colLabels=headers,
        cellLoc='center',
        loc='center',
        colWidths=[0.30, 0.35, 0.35],
        bbox=[0.05, 0.15, 0.9, 0.65]
    )
    
    # Formatação da tabela
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2.5)
    
    # Estiliza cabeçalho
    for i in range(3):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white', fontsize=13)
    
    # Alternar cores das linhas
    for i in range(1, 5):
        for j in range(3):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#F0F0F0')
    
    # Salvar
    output_path = Path('Graficos')
    output_path.mkdir(exist_ok=True)
    output_file = output_path / 'tabela_espacamentos_cristalinos.png'
    
    plt.tight_layout()
    fig.savefig(output_file, dpi=200, bbox_inches='tight')
    plt.close(fig)
    
    print(f"\n✅ Tabela salva em: {output_file}")

def print_results_summary(results_d1, results_d2):
    """
    Imprime resumo dos resultados.
    """
    print("\n" + "="*70)
    print("RESUMO - DETERMINAÇÃO DOS ESPAÇAMENTOS CRISTALINOS")
    print("="*70)
    
    print("\n📊 d₁:")
    print("-" * 70)
    print(f"  Paquímetro:")
    print(f"    d₁ = ({results_d1['paquimetro']['d']:.3f} ± {results_d1['paquimetro']['d_unc']:.3f}) Å")
    print(f"    Diferença: {abs(results_d1['paquimetro']['d'] - results_d1['d_ref'])/results_d1['d_ref'] * 100:.1f}%")
    print(f"    R²: {results_d1['paquimetro']['r_squared']:.6f}")
    
    print(f"\n  Foto:")
    print(f"    d₁ = ({results_d1['foto']['d']:.3f} ± {results_d1['foto']['d_unc']:.3f}) Å")
    print(f"    Diferença: {abs(results_d1['foto']['d'] - results_d1['d_ref'])/results_d1['d_ref'] * 100:.1f}%")
    print(f"    R²: {results_d1['foto']['r_squared']:.6f}")
    
    print("\n\n📊 d₂:")
    print("-" * 70)
    print(f"  Paquímetro:")
    print(f"    d₂ = ({results_d2['paquimetro']['d']:.3f} ± {results_d2['paquimetro']['d_unc']:.3f}) Å")
    print(f"    Diferença: {abs(results_d2['paquimetro']['d'] - results_d2['d_ref'])/results_d2['d_ref'] * 100:.1f}%")
    print(f"    R²: {results_d2['paquimetro']['r_squared']:.6f}")
    
    print(f"\n  Foto:")
    print(f"    d₂ = ({results_d2['foto']['d']:.3f} ± {results_d2['foto']['d_unc']:.3f}) Å")
    print(f"    Diferença: {abs(results_d2['foto']['d'] - results_d2['d_ref'])/results_d2['d_ref'] * 100:.1f}%")
    print(f"    R²: {results_d2['foto']['r_squared']:.6f}")
    
    print("\n\n🎯 VALORES DE REFERÊNCIA:")
    print(f"    d₁_ref = {results_d1['d_ref']:.2f} Å")
    print(f"    d₂_ref = {results_d2['d_ref']:.2f} Å")
    
    print("\n" + "="*70)

def main():
    """
    Função principal - determina espaçamentos cristalinos d₁ e d₂.
    """
    print("="*70)
    print("DETERMINAÇÃO DOS ESPAÇAMENTOS CRISTALINOS d₁ e d₂")
    print("="*70)
    
    # Verificar arquivos
    paq_path = Path("Data/finaldata_paquimetro.csv")
    foto_path = Path("Data/finaldata_computacional.csv")
    
    if not paq_path.exists():
        print(f"❌ Erro: {paq_path} não encontrado!")
        return
    
    if not foto_path.exists():
        print(f"❌ Erro: {foto_path} não encontrado!")
        return
    
    # Carregar dados
    print("\n📂 Carregando dados...")
    df_paq = pd.read_csv(paq_path)
    df_foto = pd.read_csv(foto_path)
    
    print(f"  Paquímetro: {len(df_paq)} medidas")
    print(f"  Foto: {len(df_foto)} medidas")
    
    # Gerar gráficos e determinar d₁
    print("\n📊 Gerando gráfico para d₁...")
    results_d1 = create_d_plot(df_paq, df_foto, 'r', d1_ref, 'd₁', 'espacamento_d1')
    
    # Gerar gráficos e determinar d₂
    print("\n📊 Gerando gráfico para d₂...")
    results_d2 = create_d_plot(df_paq, df_foto, 'R', d2_ref, 'd₂', 'espacamento_d2')
    
    # Gerar tabela de resultados
    print("\n📋 Gerando tabela de resultados...")
    create_d_table(results_d1, results_d2)
    
    # Resumo dos resultados
    print_results_summary(results_d1, results_d2)
    
    print("\n✅ Análise concluída!")
    print("\n📁 Arquivos gerados:")
    print("  - Graficos/espacamento_d1.png")
    print("  - Graficos/espacamento_d2.png")
    print("  - Graficos/tabela_espacamentos_cristalinos.png")
    
    return results_d1, results_d2

if __name__ == "__main__":
    results = main()
