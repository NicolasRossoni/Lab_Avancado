"""
lambda_fit.py

Análise de λ vs V^(-1/2) com ajuste linear (mínimos quadrados).
Calcula coeficiente angular e determina constante de Planck h.

Relação teórica: λ = h/√(2meV)  →  λ = (h/√(2me)) × V^(-1/2)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

# Constantes físicas
m_electron = 9.11e-31  # kg
e_charge = 1.6e-19     # C
sqrt_2me = np.sqrt(2 * m_electron * e_charge)

def linear_fit(x, y):
    """
    Ajuste linear por mínimos quadrados: y = a*x + b
    
    Returns:
        slope, intercept, slope_uncertainty, intercept_uncertainty, r_squared
    """
    # Usa scipy.stats.linregress para ajuste com incertezas
    result = stats.linregress(x, y)
    
    slope = result.slope
    intercept = result.intercept
    slope_std = result.stderr
    intercept_std = result.intercept_stderr
    r_squared = result.rvalue ** 2
    
    return slope, intercept, slope_std, intercept_std, r_squared

def create_lambda_fit_plot(df, data_source_name, output_filename):
    """
    Cria gráfico de λ vs V^(-1/2) com ajuste linear.
    
    Args:
        df: DataFrame com colunas [Volts, lambda_d1, lambda_d2]
        data_source_name: nome da fonte (e.g., "Paquímetro" ou "Foto")
        output_filename: nome do arquivo de saída (sem extensão)
    """
    # Figura maior para acomodar legenda externa
    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(111)
    
    # Cores
    colors = {
        'd1': '#E74C3C',  # Vermelho
        'd2': '#3498DB'   # Azul
    }
    
    # Calcular V^(-1/2)
    V_inv_sqrt = df['Volts'].values ** (-0.5)
    lambda_d1 = df['lambda_d1'].values
    lambda_d2 = df['lambda_d2'].values
    
    # Ajuste linear para d1
    slope_d1, intercept_d1, slope_std_d1, intercept_std_d1, r2_d1 = linear_fit(V_inv_sqrt, lambda_d1)
    
    # Ajuste linear para d2
    slope_d2, intercept_d2, slope_std_d2, intercept_std_d2, r2_d2 = linear_fit(V_inv_sqrt, lambda_d2)
    
    # Calcular h a partir do coeficiente angular
    h_d1 = slope_d1 * sqrt_2me * 1e-10  # Convertendo de Å para m
    h_d1_unc = slope_std_d1 * sqrt_2me * 1e-10
    
    h_d2 = slope_d2 * sqrt_2me * 1e-10
    h_d2_unc = slope_std_d2 * sqrt_2me * 1e-10
    
    # Gerar pontos para as retas ajustadas
    V_inv_sqrt_fit = np.linspace(V_inv_sqrt.min() * 0.95, V_inv_sqrt.max() * 1.05, 100)
    lambda_d1_fit = slope_d1 * V_inv_sqrt_fit + intercept_d1
    lambda_d2_fit = slope_d2 * V_inv_sqrt_fit + intercept_d2
    
    # Plotar dados e ajustes para d1
    ax.scatter(V_inv_sqrt, lambda_d1, color=colors['d1'], s=80, 
               marker='o', label='λ_d₁ (dados)', zorder=3, edgecolors='black', linewidth=1.5)
    ax.plot(V_inv_sqrt_fit, lambda_d1_fit, color=colors['d1'], 
            linestyle='--', linewidth=2, label='MMQ d₁', zorder=2)
    
    # Plotar dados e ajustes para d2
    ax.scatter(V_inv_sqrt, lambda_d2, color=colors['d2'], s=80,
               marker='s', label='λ_d₂ (dados)', zorder=3, edgecolors='black', linewidth=1.5)
    ax.plot(V_inv_sqrt_fit, lambda_d2_fit, color=colors['d2'],
            linestyle='--', linewidth=2, label='MMQ d₂', zorder=2)
    
    # Configurações do gráfico
    ax.set_xlabel('V⁻¹/² (V⁻¹/²)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Comprimento de Onda λ (Å)', fontsize=14, fontweight='bold')
    ax.set_title(f'Análise de λ vs V⁻¹/² - {data_source_name}',
                 fontsize=16, fontweight='bold', pad=20)
    
    # Grade
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=1)
    
    # Legenda
    ax.legend(loc='best', fontsize=11, framealpha=0.95, shadow=True)
    
    # Constante de Planck de referência
    h_ref = 6.62607015e-34  # J·s
    
    # Converter valores de h para formato (valor ± incerteza) × 10⁻³⁴
    h_d1_mantissa = h_d1 / 1e-34
    h_d1_unc_mantissa = h_d1_unc / 1e-34
    h_d2_mantissa = h_d2 / 1e-34
    h_d2_unc_mantissa = h_d2_unc / 1e-34
    h_ref_mantissa = h_ref / 1e-34
    
    # Adicionar informações dos ajustes (fora do gráfico)
    info_text = (
        f'Coeficientes Angulares:\n'
        f'd₁: a = {slope_d1:.3f} ± {slope_std_d1:.3f} Å·V¹/²\n'
        f'd₂: a = {slope_d2:.3f} ± {slope_std_d2:.3f} Å·V¹/²\n'
        f'\n'
        f'Constante de Planck:\n'
        f'h_d₁ = ({slope_d1:.3f} ± {slope_std_d1:.3f}) / √(2me)\n'
        f'    = ({h_d1_mantissa:.3f} ± {h_d1_unc_mantissa:.3f}) × 10⁻³⁴ J·s\n'
        f'h_d₂ = ({slope_d2:.3f} ± {slope_std_d2:.3f}) / √(2me)\n'
        f'    = ({h_d2_mantissa:.3f} ± {h_d2_unc_mantissa:.3f}) × 10⁻³⁴ J·s\n'
        f'\n'
        f'm = {m_electron:.2e} kg\n'
        f'e = {e_charge:.2e} C\n'
        f'h_ref = {h_ref_mantissa:.8f} × 10⁻³⁴ J·s'
    )
    
    # Colocar texto fora do gráfico (à direita)
    fig.text(0.72, 0.5, info_text, fontsize=9, verticalalignment='center',
             family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))
    
    # Salvar
    output_path = Path('Graficos')
    output_path.mkdir(exist_ok=True)
    output_file = output_path / f'{output_filename}.png'
    
    # Ajustar layout para acomodar texto externo
    plt.subplots_adjust(right=0.7)
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"\n✅ Gráfico salvo em: {output_file}")
    
    # Retornar resultados
    results = {
        'source': data_source_name,
        'd1': {
            'slope': slope_d1,
            'slope_std': slope_std_d1,
            'intercept': intercept_d1,
            'r_squared': r2_d1,
            'h': h_d1,
            'h_unc': h_d1_unc
        },
        'd2': {
            'slope': slope_d2,
            'slope_std': slope_std_d2,
            'intercept': intercept_d2,
            'r_squared': r2_d2,
            'h': h_d2,
            'h_unc': h_d2_unc
        }
    }
    
    return results

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

def create_planck_table(results_paq, results_foto):
    """
    Cria tabela com os 4 valores de h determinados.
    """
    h_ref = 6.62607015e-34  # J·s
    h_ref_mantissa = h_ref / 1e-34
    
    # Preparar dados da tabela
    table_data = []
    
    # h_d1 Paquímetro
    h_d1_paq = results_paq['d1']['h']
    h_d1_paq_unc = results_paq['d1']['h_unc']
    h_d1_paq_m = h_d1_paq / 1e-34
    h_d1_paq_unc_m = h_d1_paq_unc / 1e-34
    val_str, unc_str = format_uncertainty(h_d1_paq_m, h_d1_paq_unc_m)
    table_data.append(['h d₁ Paquímetro', f'({val_str} ± {unc_str}) × 10⁻³⁴ J·s'])
    
    # h_d2 Paquímetro
    h_d2_paq = results_paq['d2']['h']
    h_d2_paq_unc = results_paq['d2']['h_unc']
    h_d2_paq_m = h_d2_paq / 1e-34
    h_d2_paq_unc_m = h_d2_paq_unc / 1e-34
    val_str, unc_str = format_uncertainty(h_d2_paq_m, h_d2_paq_unc_m)
    table_data.append(['h d₂ Paquímetro', f'({val_str} ± {unc_str}) × 10⁻³⁴ J·s'])
    
    # h_d1 Foto
    h_d1_foto = results_foto['d1']['h']
    h_d1_foto_unc = results_foto['d1']['h_unc']
    h_d1_foto_m = h_d1_foto / 1e-34
    h_d1_foto_unc_m = h_d1_foto_unc / 1e-34
    val_str, unc_str = format_uncertainty(h_d1_foto_m, h_d1_foto_unc_m)
    table_data.append(['h d₁ Foto', f'({val_str} ± {unc_str}) × 10⁻³⁴ J·s'])
    
    # h_d2 Foto
    h_d2_foto = results_foto['d2']['h']
    h_d2_foto_unc = results_foto['d2']['h_unc']
    h_d2_foto_m = h_d2_foto / 1e-34
    h_d2_foto_unc_m = h_d2_foto_unc / 1e-34
    val_str, unc_str = format_uncertainty(h_d2_foto_m, h_d2_foto_unc_m)
    table_data.append(['h d₂ Foto', f'({val_str} ± {unc_str}) × 10⁻³⁴ J·s'])
    
    # Criar figura
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis('tight')
    ax.axis('off')
    
    # Título
    title_text = 'Determinação da Constante de Planck'
    ax.text(0.5, 0.95, title_text, ha='center', fontsize=16, weight='bold',
            transform=ax.transAxes)
    
    # Valor de referência em itálico
    ref_text = f'h_ref = {h_ref_mantissa:.8f} × 10⁻³⁴ J·s'
    ax.text(0.5, 0.88, ref_text, ha='center', fontsize=12, style='italic',
            transform=ax.transAxes)
    
    # Cabeçalhos
    headers = ['Método', 'Valor Medido']
    
    # Criar tabela
    table = ax.table(
        cellText=table_data,
        colLabels=headers,
        cellLoc='center',
        loc='center',
        colWidths=[0.35, 0.65],
        bbox=[0.1, 0.1, 0.8, 0.65]
    )
    
    # Formatação da tabela
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2.5)
    
    # Estiliza cabeçalho
    for i in range(2):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white', fontsize=13)
    
    # Alternar cores das linhas
    for i in range(1, 5):
        for j in range(2):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#F0F0F0')
    
    # Salvar
    output_path = Path('Graficos')
    output_path.mkdir(exist_ok=True)
    output_file = output_path / 'tabela_constante_planck.png'
    
    plt.tight_layout()
    fig.savefig(output_file, dpi=200, bbox_inches='tight')
    plt.close(fig)
    
    print(f"\n✅ Tabela de resultados salva em: {output_file}")

def print_results_summary(results_paq, results_foto):
    """
    Imprime resumo comparativo dos resultados.
    """
    print("\n" + "="*70)
    print("RESUMO DOS AJUSTES LINEARES E DETERMINAÇÃO DA CONSTANTE DE PLANCK")
    print("="*70)
    
    h_known = 6.626e-34  # J·s (valor aceito)
    
    print("\n📊 PAQUÍMETRO:")
    print("-" * 70)
    print(f"  d₁:")
    print(f"    Coef. angular: {results_paq['d1']['slope']:.3f} ± {results_paq['d1']['slope_std']:.3f} Å·V¹/²")
    print(f"    R²: {results_paq['d1']['r_squared']:.6f}")
    print(f"    h = ({results_paq['d1']['slope']:.3f} ± {results_paq['d1']['slope_std']:.3f}) / √(2me)")
    print(f"    h = ({results_paq['d1']['h']:.3e} ± {results_paq['d1']['h_unc']:.2e}) J·s")
    print(f"    Diferença do valor aceito: {abs(results_paq['d1']['h'] - h_known)/h_known * 100:.1f}%")
    
    print(f"\n  d₂:")
    print(f"    Coef. angular: {results_paq['d2']['slope']:.3f} ± {results_paq['d2']['slope_std']:.3f} Å·V¹/²")
    print(f"    R²: {results_paq['d2']['r_squared']:.6f}")
    print(f"    h = ({results_paq['d2']['slope']:.3f} ± {results_paq['d2']['slope_std']:.3f}) / √(2me)")
    print(f"    h = ({results_paq['d2']['h']:.3e} ± {results_paq['d2']['h_unc']:.2e}) J·s")
    print(f"    Diferença do valor aceito: {abs(results_paq['d2']['h'] - h_known)/h_known * 100:.1f}%")
    
    print("\n\n📊 FOTO (COMPUTACIONAL):")
    print("-" * 70)
    print(f"  d₁:")
    print(f"    Coef. angular: {results_foto['d1']['slope']:.3f} ± {results_foto['d1']['slope_std']:.3f} Å·V¹/²")
    print(f"    R²: {results_foto['d1']['r_squared']:.6f}")
    print(f"    h = ({results_foto['d1']['slope']:.3f} ± {results_foto['d1']['slope_std']:.3f}) / √(2me)")
    print(f"    h = ({results_foto['d1']['h']:.3e} ± {results_foto['d1']['h_unc']:.2e}) J·s")
    print(f"    Diferença do valor aceito: {abs(results_foto['d1']['h'] - h_known)/h_known * 100:.1f}%")
    
    print(f"\n  d₂:")
    print(f"    Coef. angular: {results_foto['d2']['slope']:.3f} ± {results_foto['d2']['slope_std']:.3f} Å·V¹/²")
    print(f"    R²: {results_foto['d2']['r_squared']:.6f}")
    print(f"    h = ({results_foto['d2']['slope']:.3f} ± {results_foto['d2']['slope_std']:.3f}) / √(2me)")
    print(f"    h = ({results_foto['d2']['h']:.3e} ± {results_foto['d2']['h_unc']:.2e}) J·s")
    print(f"    Diferença do valor aceito: {abs(results_foto['d2']['h'] - h_known)/h_known * 100:.1f}%")
    
    print(f"\n\n🎯 VALOR ACEITO DA CONSTANTE DE PLANCK:")
    print(f"    h = {h_known:.3e} J·s")
    
    print("\n" + "="*70)

def main():
    """
    Função principal - carrega dados e gera gráficos de ajuste.
    """
    print("="*70)
    print("ANÁLISE DE λ vs V⁻¹/² - DETERMINAÇÃO DA CONSTANTE DE PLANCK")
    print("="*70)
    
    # Verificar arquivos
    paq_path = Path("Data/lambda_analysis_paquimetro_final.csv")
    foto_path = Path("Data/lambda_analysis_computacional_final.csv")
    
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
    
    # Gerar gráficos e ajustes
    print("\n📊 Gerando gráficos e ajustes lineares...")
    results_paq = create_lambda_fit_plot(df_paq, "Paquímetro", "ajuste_linear_paquimetro")
    results_foto = create_lambda_fit_plot(df_foto, "Foto (Computacional)", "ajuste_linear_foto")
    
    # Gerar tabela de resultados
    print("\n📋 Gerando tabela de resultados...")
    create_planck_table(results_paq, results_foto)
    
    # Resumo dos resultados
    print_results_summary(results_paq, results_foto)
    
    print("\n✅ Análise concluída!")
    print("\n📁 Arquivos gerados:")
    print("  - Graficos/ajuste_linear_paquimetro.png")
    print("  - Graficos/ajuste_linear_foto.png")
    print("  - Graficos/tabela_constante_planck.png")
    
    return results_paq, results_foto

if __name__ == "__main__":
    results = main()
