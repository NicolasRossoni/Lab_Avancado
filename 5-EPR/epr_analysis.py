#!/usr/bin/env python3
"""
epr_analysis.py

Análise de Ressonância Paramagnética Eletrônica (EPR) - Experimento 5-EPR

Determina o fator g do composto dFPH através da relação:
ν = (g * μ / h) * B

Onde:
- ν = frequência de ressonância [Hz]
- B = campo magnético [T]
- h = constante de Planck = 6.62607015 × 10⁻³⁴ J·s
- μ = magneton de Bohr = 9.2740100783 × 10⁻²⁴ J/T
- g = fator g do dFPH (valor a determinar)

O ajuste linear de ν vs B fornece o coeficiente angular a = g*μ/h,
permitindo calcular g = a*h/μ com sua incerteza.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

# =========================
# CONSTANTES FÍSICAS
# =========================

# Constante de Planck [J·s]
h_planck = 6.62607015e-34

# Magneton de Bohr [J/T]
# NOTA: Deixado como variável ajustável caso necessário
mu_B = 9.2740100783e-24

# Parâmetros da bobina de Helmholtz (do sonda_hall.py)
mu_0 = 4 * np.pi * 1e-7  # Permeabilidade do vácuo [Vs/Am]
n = 320  # Número de espiras
r = 6.8e-2  # Raio da bobina em metros (6.8 cm)
geometric_factor = (4/5)**(3/2)  # Fator geométrico Helmholtz

# Constante de conversão corrente -> campo magnético [T/A]
K_helmholtz = mu_0 * geometric_factor * n / r

print("="*70)
print("ANÁLISE EPR - FATOR g DO COMPOSTO dFPH")
print("="*70)
print(f"Constantes físicas:")
print(f"  h (Planck)      = {h_planck:.4e} J·s")
print(f"  μ_B (Bohr)      = {mu_B:.4e} J/T")
print(f"\nParâmetros da bobina de Helmholtz:")
print(f"  μ₀              = {mu_0:.4e} Vs/Am")
print(f"  n               = {n} espiras")
print(f"  r               = {r*100:.1f} cm")
print(f"  K (I→B)         = {K_helmholtz:.4e} T/A")
print("="*70)


def current_to_field(I_amperes):
    """
    Converte corrente [A] para campo magnético [T] usando fórmula de Helmholtz.

    B = μ₀ * (4/5)^(3/2) * n/r * I

    Args:
        I_amperes: corrente em Ampères

    Returns:
        campo magnético em Tesla
    """
    return K_helmholtz * I_amperes


def load_frequency_data(filename):
    """
    Carrega arquivo CSV com dados de frequência vs corrente.

    Formato esperado: ν (MHz), I (A) com vírgula como separador decimal.

    Args:
        filename: nome do arquivo CSV

    Returns:
        DataFrame com colunas freq_MHz, I_A, freq_Hz, B_T
    """
    filepath = Path("Data") / filename

    # Lê CSV com vírgula como separador decimal (formato brasileiro)
    df = pd.read_csv(filepath, decimal=',')

    # Renomeia colunas para facilitar acesso
    df.columns = ['freq_MHz', 'I_A']

    # Conversões
    df['freq_Hz'] = df['freq_MHz'] * 1e6  # MHz -> Hz
    df['B_T'] = current_to_field(df['I_A'])  # A -> T
    df['B_mT'] = df['B_T'] * 1000  # T -> mT (para display)

    return df


def linear_fit_with_uncertainty(x, y):
    """
    Ajuste linear por mínimos quadrados com cálculo de incertezas.

    Modelo: y = a * x + b

    Args:
        x: variável independente (campo magnético B)
        y: variável dependente (frequência ν)

    Returns:
        tuple: (slope, intercept, slope_uncertainty, intercept_uncertainty, r_squared)
    """
    # Regressão linear usando scipy.stats
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

    # Cálculo manual da incerteza do coeficiente angular
    n = len(x)
    y_pred = slope * x + intercept
    residuals = y - y_pred

    # Soma dos quadrados dos resíduos
    ss_res = np.sum(residuals**2)

    # Variância dos resíduos
    s_squared = ss_res / (n - 2)

    # Incerteza do coeficiente angular
    x_mean = np.mean(x)
    ss_x = np.sum((x - x_mean)**2)
    slope_uncertainty = np.sqrt(s_squared / ss_x)

    # Incerteza do intercepto
    intercept_uncertainty = np.sqrt(s_squared * (1/n + x_mean**2/ss_x))

    # Coeficiente de determinação
    r_squared = r_value**2

    return slope, intercept, slope_uncertainty, intercept_uncertainty, r_squared


def calculate_g_factor(slope, slope_uncertainty):
    """
    Calcula o fator g e sua incerteza a partir do coeficiente angular.

    Da relação ν = (g*μ_B/h)*B, temos:
    - Coeficiente angular: a = g*μ_B/h
    - Fator g: g = a*h/μ_B
    - Incerteza: δg = δa * h/μ_B

    Args:
        slope: coeficiente angular [Hz/T]
        slope_uncertainty: incerteza do coeficiente angular [Hz/T]

    Returns:
        tuple: (g_factor, g_uncertainty)
    """
    g_factor = (slope * h_planck) / mu_B
    g_uncertainty = (slope_uncertainty * h_planck) / mu_B

    return g_factor, g_uncertainty


def format_value_with_uncertainty(value, uncertainty):
    """
    Formata valor com incerteza seguindo regras de algarismos significativos.

    Regras:
    1. Arredonda incerteza para 1 algarismo significativo
    2. Arredonda valor para o mesmo decimal da incerteza

    Args:
        value: valor central
        uncertainty: incerteza

    Returns:
        tuple: (formatted_value, formatted_uncertainty, decimals)
    """
    # Ordem de grandeza da incerteza
    if uncertainty == 0:
        return f"{value:.6f}", "0", 6

    # Encontra primeiro algarismo significativo da incerteza
    uncertainty_order = int(np.floor(np.log10(abs(uncertainty))))

    # Arredonda incerteza para 1 alg. sig.
    uncertainty_rounded = round(uncertainty, -uncertainty_order)

    # Número de casas decimais
    decimals = max(0, -uncertainty_order)

    # Arredonda valor para mesma precisão
    value_rounded = round(value, decimals)

    # Formata strings
    if decimals == 0:
        value_str = f"{int(value_rounded)}"
        unc_str = f"{int(uncertainty_rounded)}"
    else:
        value_str = f"{value_rounded:.{decimals}f}"
        unc_str = f"{uncertainty_rounded:.{decimals}f}"

    return value_str, unc_str, decimals


def analyze_frequency_file(filename):
    """
    Analisa um arquivo de dados de frequência.

    Args:
        filename: nome do arquivo CSV

    Returns:
        dict: resultados da análise (df, slope, intercept, g, g_unc, r2)
    """
    print(f"\n{'─'*70}")
    print(f"Analisando: {filename}")
    print(f"{'─'*70}")

    # Carrega dados
    df = load_frequency_data(filename)

    print(f"Dados carregados: {len(df)} pontos")
    print(f"  Frequência: {df['freq_MHz'].min():.1f} - {df['freq_MHz'].max():.1f} MHz")
    print(f"  Corrente:   {df['I_A'].min():.3f} - {df['I_A'].max():.3f} A")
    print(f"  Campo B:    {df['B_mT'].min():.2f} - {df['B_mT'].max():.2f} mT")

    # Ajuste linear: ν vs B
    slope, intercept, slope_unc, intercept_unc, r2 = linear_fit_with_uncertainty(
        df['B_T'].values,
        df['freq_Hz'].values
    )

    print(f"\nAjuste linear (ν = a·B + b):")
    print(f"  Coef. angular (a): {slope:.4e} ± {slope_unc:.2e} Hz/T")
    print(f"  Intercepto (b):    {intercept:.4e} ± {intercept_unc:.2e} Hz")
    print(f"  R²:                {r2:.8f}")

    # Calcula fator g
    g_factor, g_uncertainty = calculate_g_factor(slope, slope_unc)

    print(f"\nFator g(dFPH):")
    print(f"  g = {g_factor:.6f} ± {g_uncertainty:.6f}")

    # Formata com algarismos significativos
    g_str, g_unc_str, decimals = format_value_with_uncertainty(g_factor, g_uncertainty)
    print(f"  g = {g_str} ± {g_unc_str}")

    return {
        'filename': filename,
        'df': df,
        'slope': slope,
        'intercept': intercept,
        'slope_unc': slope_unc,
        'intercept_unc': intercept_unc,
        'r2': r2,
        'g_factor': g_factor,
        'g_uncertainty': g_uncertainty,
        'g_formatted': g_str,
        'g_unc_formatted': g_unc_str,
        'decimals': decimals
    }


def create_epr_plot(results, output_dir="Graficos"):
    """
    Cria gráfico de EPR com scatter plot e ajuste linear.

    Args:
        results: dicionário com resultados da análise
        output_dir: diretório para salvar gráfico
    """
    # Extrai dados
    df = results['df']
    filename = results['filename']
    slope = results['slope']
    intercept = results['intercept']
    g_factor = results['g_factor']
    g_str = results['g_formatted']
    g_unc_str = results['g_unc_formatted']
    r2 = results['r2']

    # Determina faixa de frequência para título
    freq_range = filename.replace('.csv', '').replace('_', '-')

    # Cria figura
    fig, ax = plt.subplots(figsize=(10, 8))

    # Cores
    color_data = '#2E86AB'    # Azul para dados
    color_fit = '#A23B72'     # Roxo para ajuste

    # Scatter plot dos dados experimentais
    ax.scatter(df['B_mT'], df['freq_MHz'],
              color=color_data, s=100, alpha=0.8,
              label='Dados Experimentais', zorder=3,
              edgecolors='white', linewidths=1.5)

    # Linha do ajuste linear
    B_fit = np.linspace(df['B_mT'].min() * 0.95, df['B_mT'].max() * 1.05, 200)
    B_fit_T = B_fit / 1000  # mT -> T
    freq_fit_Hz = slope * B_fit_T + intercept
    freq_fit_MHz = freq_fit_Hz / 1e6  # Hz -> MHz

    ax.plot(B_fit, freq_fit_MHz,
           color=color_fit, linewidth=3, alpha=0.9,
           label=f'Ajuste Linear (R² = {r2:.6f})', zorder=2)

    # Formatação
    ax.set_xlabel('Campo Magnético B (mT)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Frequência ν (MHz)', fontsize=14, fontweight='bold')

    title = f'EPR: Medindo g do composto dFPH\n({freq_range})'
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)

    # Grid
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.8)
    ax.set_axisbelow(True)

    # Legenda
    legend = ax.legend(fontsize=12, loc='upper left',
                      frameon=True, fancybox=True, shadow=True)
    legend.get_frame().set_alpha(0.9)

    # Texto com resultado do g
    g_text = (
        f"g(dFPH) = h·ν / (μ_B·B)\n"
        f"\n"
        f"g = {g_str} ± {g_unc_str}"
    )

    ax.text(0.97, 0.05, g_text, transform=ax.transAxes,
           fontsize=13, verticalalignment='bottom', horizontalalignment='right',
           bbox=dict(boxstyle="round,pad=0.6", facecolor="lightgray", alpha=0.85),
           fontweight='bold')

    # Formatação dos ticks
    ax.tick_params(axis='both', which='major', labelsize=11)

    # Layout
    plt.tight_layout()

    # Salva gráfico
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    output_file = output_path / f"epr_{freq_range}.png"

    fig.savefig(output_file, dpi=300, bbox_inches='tight',
               facecolor='white', edgecolor='none')

    print(f"\nGráfico salvo: {output_file}")

    plt.close(fig)

    return output_file


def main():
    """
    Função principal - processa todos os arquivos de frequência.
    """
    print("\nIniciando análise EPR...\n")

    # Arquivos de dados
    frequency_files = [
        "15_30_MHz.csv",
        "30_60_MHz.csv",
        "45_90_MHz.csv"
    ]

    # Analisa cada arquivo
    all_results = []

    for filename in frequency_files:
        # Verifica se arquivo existe
        filepath = Path("Data") / filename
        if not filepath.exists():
            print(f"AVISO: Arquivo {filepath} não encontrado. Pulando...")
            continue

        # Analisa dados
        results = analyze_frequency_file(filename)
        all_results.append(results)

        # Gera gráfico
        create_epr_plot(results)

    # Resumo final
    print(f"\n{'='*70}")
    print("RESUMO FINAL - FATOR g(dFPH)")
    print(f"{'='*70}")

    if len(all_results) > 0:
        print(f"\n{'Arquivo':<20} {'g':<15} {'Incerteza':<15} {'R²':<10}")
        print(f"{'-'*60}")

        g_values = []
        for res in all_results:
            print(f"{res['filename']:<20} {res['g_factor']:<15.6f} {res['g_uncertainty']:<15.6f} {res['r2']:<10.8f}")
            g_values.append(res['g_factor'])

        # Estatísticas dos valores de g
        g_mean = np.mean(g_values)
        g_std = np.std(g_values)

        print(f"\nEstatísticas dos valores de g:")
        print(f"  Média: {g_mean:.6f}")
        print(f"  Desvio padrão: {g_std:.6f}")
        print(f"  Consistência: {'Excelente' if g_std < 0.01 else 'Boa' if g_std < 0.05 else 'Regular'}")

        print(f"\n{'='*70}")
        print(f"Análise concluída! {len(all_results)} arquivos processados.")
        print(f"Gráficos salvos em: Graficos/")
        print(f"{'='*70}")

    return all_results


if __name__ == "__main__":
    results = main()
