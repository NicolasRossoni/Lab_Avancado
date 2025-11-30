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

# Incertezas experimentais (baseadas nos últimos algarismos significativos)
n_uncertainty = 1  # espiras (último alg. sig.)
r_uncertainty = 0.001  # m (0.1 cm convertido para metros)
I_uncertainty = 0.01  # A (último alg. sig. da corrente)
freq_uncertainty_MHz = 0.1  # MHz (último alg. sig. da frequência nos dados)

print("="*70)
print("ANÁLISE EPR - FATOR g DO COMPOSTO dFPH")
print("="*70)
print(f"Constantes físicas:")
print(f"  h (Planck)      = {h_planck:.4e} J·s")
print(f"  μ_B (Bohr)      = {mu_B:.4e} J/T")
print(f"\nParâmetros da bobina de Helmholtz:")
print(f"  μ₀              = {mu_0:.4e} Vs/Am")
print(f"  n               = {n} ± {n_uncertainty} espiras")
print(f"  r               = {r*100:.1f} ± {r_uncertainty*100:.1f} cm")
print(f"  K (I→B)         = {K_helmholtz:.4e} T/A")
print(f"\nIncertezas experimentais:")
print(f"  δI              = {I_uncertainty} A")
print(f"  δν              = {freq_uncertainty_MHz} MHz")
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


def calculate_field_uncertainty(I_values):
    """
    Calcula incerteza do campo magnético usando propagação de erros.

    B = μ₀ * (4/5)^(3/2) * n/r * I

    Usando propagação de erros para erros relativos:
    (δB/B)² = (δn/n)² + (δr/r)² + (δI/I)²

    Args:
        I_values: array com valores de corrente em Ampères

    Returns:
        array: incerteza do campo magnético em Tesla
    """
    # Calcula erro relativo de cada componente
    relative_n = n_uncertainty / n
    relative_r = r_uncertainty / r

    # Para cada valor de corrente, calcula o erro relativo
    I_values = np.atleast_1d(I_values)  # Garante que é array

    # Evita divisão por zero
    with np.errstate(divide='ignore', invalid='ignore'):
        relative_I = np.where(I_values != 0, I_uncertainty / I_values, 0)

    # Erro relativo total
    relative_error = np.sqrt(relative_n**2 + relative_r**2 + relative_I**2)

    # Calcula campo magnético
    B_values = current_to_field(I_values)

    # Incerteza absoluta em Tesla
    B_uncertainty = B_values * relative_error

    return B_uncertainty


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
    color_fit = '#FDB30E'     # Amarelo para ajuste

    # Incertezas experimentais
    freq_unc_MHz = freq_uncertainty_MHz  # MHz (último alg. sig.)

    # Calcula incerteza do campo magnético usando propagação de erros
    B_uncertainty_T = calculate_field_uncertainty(df['I_A'].values)
    B_uncertainty_mT = B_uncertainty_T * 1000  # T -> mT

    # Scatter plot dos dados experimentais com barras de erro
    ax.errorbar(df['B_mT'], df['freq_MHz'],
               xerr=B_uncertainty_mT, yerr=freq_unc_MHz,
               fmt='o', color=color_data, markersize=9, alpha=0.8,
               label='Dados Experimentais', zorder=3,
               markeredgecolor='white', markeredgewidth=1.5,
               elinewidth=1.5, capsize=3, capthick=1.5)

    # Linha do ajuste linear
    B_fit = np.linspace(df['B_mT'].min() * 0.95, df['B_mT'].max() * 1.05, 200)
    B_fit_T = B_fit / 1000  # mT -> T
    freq_fit_Hz = slope * B_fit_T + intercept
    freq_fit_MHz = freq_fit_Hz / 1e6  # Hz -> MHz

    ax.plot(B_fit, freq_fit_MHz,
           color=color_fit, linewidth=3, alpha=0.9,
           label='Ajuste Linear', zorder=2)

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


def create_g_table(all_results, output_dir="Graficos"):
    """
    Cria tabela com valores de g para cada frequência e média.

    Args:
        all_results: lista de dicionários com resultados de cada análise
        output_dir: diretório para salvar a tabela
    """
    # Cria diretório se não existir
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Prepara dados da tabela
    table_data = []
    g_values = []

    for res in all_results:
        # Extrai faixa de frequência do nome do arquivo
        freq_range = res['filename'].replace('.csv', '').replace('_', '-')

        # Formata g e incerteza
        g_str, g_unc_str, _ = format_value_with_uncertainty(
            res['g_factor'], res['g_uncertainty']
        )

        table_data.append([
            freq_range,
            f"{g_str} ± {g_unc_str}"
        ])

        g_values.append(res['g_factor'])

    # Calcula média e desvio padrão
    g_mean = np.mean(g_values)
    g_std = np.std(g_values, ddof=1)  # Desvio padrão amostral

    # Formata média com arredondamento correto
    g_mean_str, g_std_str, _ = format_value_with_uncertainty(g_mean, g_std)

    # Cria figura para tabela
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('tight')
    ax.axis('off')

    # Título
    title_text = "Fator g do composto dFPH - Análise EPR"
    ax.set_title(title_text, fontsize=16, weight='bold', pad=30)

    # Cabeçalhos
    headers = [
        "Faixa de Frequência",
        "Fator g ± Incerteza"
    ]

    # Cria tabela
    table = ax.table(
        cellText=table_data,
        colLabels=headers,
        cellLoc='center',
        loc='center',
        colWidths=[0.45, 0.45]
    )

    # Formatação da tabela
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2.5)

    # Estiliza cabeçalho (azul como no 4-Eletron)
    for i in range(2):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white', fontsize=13)

    # Adiciona média abaixo da tabela (fora da tabela, em preto, negrito)
    mean_text = f"Média: g = {g_mean_str} ± {g_std_str}"
    ax.text(0.5, 0.25, mean_text, ha='center', fontsize=14,
            weight='bold', color='black', transform=ax.transAxes)

    # Adiciona valor literário (referência)
    g_lit = 2.0036  # Valor do g do elétron livre (aproximadamente)
    lit_text = f"g_literatura = {g_lit:.4f} (Elétron livre - Referência)"
    ax.text(0.5, 0.15, lit_text, ha='center', fontsize=12,
            style='italic', color='#555555', transform=ax.transAxes)

    # Salva tabela
    output_file = output_path / "tabela_fator_g.png"
    fig.tight_layout()
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"\nTabela de fatores g salva: {output_file}")

    return output_file


def create_combined_epr_plot(all_results, output_dir="Graficos"):
    """
    Cria gráfico EPR combinado com todos os pontos de todas as frequências.

    Usa cores diferentes para cada faixa de frequência, mas um único ajuste linear.

    Args:
        all_results: lista de dicionários com resultados de cada análise
        output_dir: diretório para salvar o gráfico
    """
    # Combina todos os DataFrames
    all_data = []
    colors_map = {
        '15_30_MHz.csv': '#E63946',    # Vermelho
        '30_60_MHz.csv': '#2E86AB',    # Azul
        '45_90_MHz.csv': '#06A77D'     # Verde
    }

    labels_map = {
        '15_30_MHz.csv': '15-30 MHz',
        '30_60_MHz.csv': '30-60 MHz',
        '45_90_MHz.csv': '45-90 MHz'
    }

    for res in all_results:
        df = res['df'].copy()
        df['filename'] = res['filename']
        df['color'] = colors_map.get(res['filename'], '#000000')
        df['label'] = labels_map.get(res['filename'], res['filename'])
        all_data.append(df)

    # Concatena todos os dados
    df_combined = pd.concat(all_data, ignore_index=True)

    print(f"\n{'─'*70}")
    print(f"Criando gráfico combinado com todos os pontos")
    print(f"{'─'*70}")
    print(f"Total de pontos: {len(df_combined)}")

    # Faz ajuste linear com TODOS os pontos
    slope, intercept, slope_unc, intercept_unc, r2 = linear_fit_with_uncertainty(
        df_combined['B_T'].values,
        df_combined['freq_Hz'].values
    )

    print(f"\nAjuste linear combinado (ν = a·B + b):")
    print(f"  Coef. angular (a): {slope:.4e} ± {slope_unc:.2e} Hz/T")
    print(f"  Intercepto (b):    {intercept:.4e} ± {intercept_unc:.2e} Hz")
    print(f"  R²:                {r2:.8f}")

    # Calcula fator g combinado
    g_combined, g_combined_unc = calculate_g_factor(slope, slope_unc)
    g_combined_str, g_combined_unc_str, _ = format_value_with_uncertainty(g_combined, g_combined_unc)

    print(f"\nFator g combinado:")
    print(f"  g_médio = {g_combined_str} ± {g_combined_unc_str}")

    # Cria figura
    fig, ax = plt.subplots(figsize=(12, 9))

    color_fit = '#FDB30E'  # Amarelo para ajuste

    # Calcula incertezas
    freq_unc_MHz = freq_uncertainty_MHz
    B_uncertainty_T = calculate_field_uncertainty(df_combined['I_A'].values)
    B_uncertainty_mT = B_uncertainty_T * 1000

    # Plot com cores diferentes para cada arquivo
    for filename in df_combined['filename'].unique():
        df_subset = df_combined[df_combined['filename'] == filename]
        color = df_subset['color'].iloc[0]
        label = df_subset['label'].iloc[0]

        # Índices do subset no df_combined
        indices = df_combined[df_combined['filename'] == filename].index

        ax.errorbar(df_subset['B_mT'].values, df_subset['freq_MHz'].values,
                   xerr=B_uncertainty_mT[indices], yerr=freq_unc_MHz,
                   fmt='o', color=color, markersize=8, alpha=0.8,
                   label=label, zorder=3,
                   markeredgecolor='white', markeredgewidth=1.5,
                   elinewidth=1.5, capsize=3, capthick=1.5)

    # Linha do ajuste linear (única, para todos os pontos)
    B_fit = np.linspace(df_combined['B_mT'].min() * 0.95, df_combined['B_mT'].max() * 1.05, 200)
    B_fit_T = B_fit / 1000  # mT -> T
    freq_fit_Hz = slope * B_fit_T + intercept
    freq_fit_MHz = freq_fit_Hz / 1e6  # Hz -> MHz

    ax.plot(B_fit, freq_fit_MHz,
           color=color_fit, linewidth=3, alpha=0.9,
           label='Ajuste Linear Combinado', zorder=2)

    # Formatação
    ax.set_xlabel('Campo Magnético B (mT)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Frequência ν (MHz)', fontsize=14, fontweight='bold')

    title = f'EPR: Análise Combinada - Fator g do dFPH'
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)

    # Grid
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.8)
    ax.set_axisbelow(True)

    # Legenda
    legend = ax.legend(fontsize=11, loc='upper left',
                      frameon=True, fancybox=True, shadow=True)
    legend.get_frame().set_alpha(0.9)

    # Formatação dos ticks
    ax.tick_params(axis='both', which='major', labelsize=11)

    # Layout
    plt.tight_layout()

    # Salva gráfico
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    output_file = output_path / "epr_combined_all_frequencies.png"

    fig.savefig(output_file, dpi=300, bbox_inches='tight',
               facecolor='white', edgecolor='none')

    print(f"\nGráfico combinado salvo: {output_file}")

    plt.close(fig)

    return output_file, g_combined, g_combined_unc


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

        # Gera tabela com fatores g
        print(f"\nGerando tabela com fatores g...")
        create_g_table(all_results)

        # Gera gráfico combinado com todos os pontos
        print(f"\nGerando gráfico combinado com todos os pontos...")
        combined_file, g_combined, g_combined_unc = create_combined_epr_plot(all_results)

        print(f"\n{'='*70}")
        print(f"Análise concluída! {len(all_results)} arquivos processados.")
        print(f"Gráficos individuais, gráfico combinado e tabela salvos em: Graficos/")
        print(f"\nResultado do ajuste combinado:")
        print(f"  g_médio = {g_combined:.6f} ± {g_combined_unc:.6f}")
        print(f"{'='*70}")

    return all_results


if __name__ == "__main__":
    results = main()
