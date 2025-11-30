#!/usr/bin/env python3
"""
sonda_hall.py

Análise do Campo Magnético da Sonda Hall no Experimento 5-EPR

Compara dados experimentais (I vs B) com a curva teórica:
B = μ₀ * (4/5)^(3/2) * n/r * I

Onde:
- μ₀ = 4π × 10⁻⁷ [Vs/Am] (permeabilidade do vácuo)
- n = 320 (número de espiras)
- r = 6.8 cm = 0.068 m (raio da bobina)
- I = corrente em Ampères
- B = campo magnético em Tesla
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# =========================
# CONSTANTES EXPERIMENTAIS
# =========================

# Parâmetros da bobina de Helmholtz
mu_0 = 4 * np.pi * 1e-7  # Permeabilidade do vácuo [Vs/Am]
n = 320  # Número de espiras
r = 6.8e-2  # Raio da bobina em metros (6.8 cm)

# Fator geométrico para bobinas de Helmholtz
geometric_factor = (4/5)**(3/2)  # ≈ 0.7155

# Constante teórica B = K * I
K_theoretical = mu_0 * geometric_factor * n / r  # [T/A]

# Incertezas experimentais (baseadas nos últimos algarismos significativos)
n_uncertainty = 1  # espiras (último alg. sig.)
r_uncertainty = 0.001  # m (0.1 cm convertido para metros)
I_uncertainty = 0.01  # A (último alg. sig. da corrente)

print(f"="*60)
print(f"PARÂMETROS DO EXPERIMENTO 5-EPR - SONDA HALL")
print(f"="*60)
print(f"μ₀ = {mu_0:.2e} Vs/Am")
print(f"n = {n} ± {n_uncertainty} espiras")
print(f"r = {r*100:.1f} ± {r_uncertainty*100:.1f} cm = {r:.3f} ± {r_uncertainty:.3f} m")
print(f"Fator geométrico (4/5)^(3/2) = {geometric_factor:.4f}")
print(f"Constante teórica K = μ₀ * (4/5)^(3/2) * n/r = {K_theoretical:.2e} T/A")
print(f"="*60)

def load_experimental_data():
    """
    Carrega dados experimentais do arquivo sonda_hall.csv.
    
    Returns:
        DataFrame: dados com colunas I (A) e B (mT)
    """
    data_path = "Data/sonda_hall.csv"
    
    # Carrega dados
    df = pd.read_csv(data_path)
    
    # Renomeia colunas para facilitar o acesso
    df.columns = ['I_A', 'B_mT']
    
    # Converte campo magnético de mT para T (Tesla)
    df['B_T'] = df['B_mT'] / 1000.0  # mT -> T
    
    print(f"\nDados experimentais carregados:")
    print(f"- {len(df)} pontos de medição")
    print(f"- Corrente: {df['I_A'].min():.2f} - {df['I_A'].max():.2f} A")
    print(f"- Campo magnético: {df['B_mT'].min():.2f} - {df['B_mT'].max():.2f} mT")
    
    return df

def calculate_theoretical_field(I_values):
    """
    Calcula campo magnético teórico usando a fórmula das bobinas de Helmholtz.
    
    B = μ₀ * (4/5)^(3/2) * n/r * I
    
    Args:
        I_values: array com valores de corrente em Ampères
    
    Returns:
        array: campo magnético teórico em Tesla
    """
    B_theoretical = K_theoretical * I_values
    return B_theoretical

def calculate_theoretical_field_mT(I_values):
    """
    Calcula campo magnético teórico em miliTesla.

    Args:
        I_values: array com valores de corrente em Ampères

    Returns:
        array: campo magnético teórico em miliTesla
    """
    return calculate_theoretical_field(I_values) * 1000.0  # T -> mT

def calculate_field_uncertainty(I_values):
    """
    Calcula incerteza do campo magnético usando propagação de erros.

    B = μ₀ * (4/5)^(3/2) * n/r * I

    Usando propagação de erros para erros relativos:
    (δB/B)² = (δn/n)² + (δr/r)² + (δI/I)²

    Args:
        I_values: array com valores de corrente em Ampères

    Returns:
        array: incerteza do campo magnético em miliTesla
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
    B_values = calculate_theoretical_field(I_values)

    # Incerteza absoluta em Tesla
    B_uncertainty = B_values * relative_error

    # Converte para miliTesla
    return B_uncertainty * 1000.0

def analyze_data(df):
    """
    Analisa concordância entre dados experimentais e teoria.
    
    Args:
        df: DataFrame com dados experimentais
    """
    print(f"\n{'-'*50}")
    print(f"ANÁLISE EXPERIMENTAL VS TEÓRICA")
    print(f"{'-'*50}")
    
    # Calcula valores teóricos para as correntes experimentais
    B_theoretical_mT = calculate_theoretical_field_mT(df['I_A'])
    
    # Calcula diferenças
    diff_mT = df['B_mT'] - B_theoretical_mT
    diff_percent = (diff_mT / B_theoretical_mT) * 100
    
    print(f"\nComparação ponto a ponto:")
    print(f"{'I (A)':>6} {'B_exp (mT)':>12} {'B_teo (mT)':>12} {'Diff (mT)':>11} {'Diff (%)':>9}")
    print(f"{'-'*52}")
    
    for i, row in df.iterrows():
        I = row['I_A']
        B_exp = row['B_mT']
        B_teo = B_theoretical_mT[i]
        diff = diff_mT[i]
        diff_pct = diff_percent[i]
        
        print(f"{I:6.2f} {B_exp:12.2f} {B_teo:12.2f} {diff:11.2f} {diff_pct:8.1f}%")
    
    # Estatísticas resumo
    mean_diff = np.mean(np.abs(diff_mT))
    max_diff = np.max(np.abs(diff_mT))
    mean_diff_pct = np.mean(np.abs(diff_percent))
    
    print(f"\nEstatísticas de diferença:")
    print(f"Diferença média absoluta: {mean_diff:.2f} mT ({mean_diff_pct:.1f}%)")
    print(f"Diferença máxima absoluta: {max_diff:.2f} mT")
    
    # Calcula coeficiente angular experimental
    # B = K_exp * I (regressão linear forçada pela origem)
    K_experimental = np.sum(df['I_A'] * df['B_T']) / np.sum(df['I_A']**2)  # em T/A
    
    print(f"\nCoeficientes angulares:")
    print(f"K_teórico = {K_theoretical:.4e} T/A")
    print(f"K_experimental = {K_experimental:.4e} T/A")
    print(f"Razão K_exp/K_teo = {K_experimental/K_theoretical:.3f}")
    
    print(f"{'-'*50}")
    
    return B_theoretical_mT, K_experimental

def create_hall_plot(df, output_dir="Graficos"):
    """
    Cria gráfico no estilo do 4-Eletron com scatter plot e curva teórica.
    
    Args:
        df: DataFrame com dados experimentais
        output_dir: diretório para salvar o gráfico
    """
    # Cria diretório se não existir
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Configura figura no estilo do 4-Eletron
    plt.style.use('default')  # Reset para estilo padrão
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Cores no estilo do 4-Eletron
    color_exp = '#2E86AB'      # Azul para dados experimentais
    color_theory = '#FDB30E'   # Amarelo para curva teórica

    # Incertezas dos dados experimentais
    I_uncertainty_A = I_uncertainty  # Última casa decimal da corrente (0.01 A)

    # Calcula incerteza do campo magnético usando propagação de erros
    B_uncertainty_mT = calculate_field_uncertainty(df['I_A'].values)

    # Scatter plot dos dados experimentais com barras de erro
    ax.errorbar(df['I_A'], df['B_mT'],
               yerr=B_uncertainty_mT, xerr=I_uncertainty_A,
               fmt='o', color=color_exp, markersize=8, alpha=0.8,
               label='Dados Experimentais', zorder=3,
               markeredgecolor='white', markeredgewidth=1.5,
               elinewidth=1.5, capsize=3, capthick=1.5)
    
    # Curva teórica (mais pontos para suavidade)
    I_smooth = np.linspace(0, df['I_A'].max() * 1.1, 200)
    B_theory_smooth = calculate_theoretical_field_mT(I_smooth)
    
    ax.plot(I_smooth, B_theory_smooth,
           color=color_theory, linewidth=3, alpha=0.9,
           label='Curva Teórica', zorder=2)
    
    # Formatação do gráfico no estilo do 4-Eletron
    ax.set_xlabel('Corrente I (A)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Campo Magnético B (mT)', fontsize=14, fontweight='bold')
    ax.set_title('Experimento 5-EPR: Sonda Hall\nCampo Magnético vs Corrente', 
                fontsize=16, fontweight='bold', pad=20)
    
    # Grid no estilo do 4-Eletron
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.8)
    ax.set_axisbelow(True)
    
    # Legenda estilizada
    legend = ax.legend(fontsize=12, loc='upper left', 
                      frameon=True, fancybox=True, shadow=True)
    legend.get_frame().set_alpha(0.9)
    
    # Adiciona informações dos parâmetros
    info_text = (
        f"Parâmetros:\n"
        f"μ₀ = {mu_0:.2e} Vs/Am\n"
        f"n = {n} espiras\n"
        f"r = {r*100:.1f} cm\n"
        f"Fator = (4/5)^(3/2) = {geometric_factor:.4f}"
    )
    
    ax.text(0.98, 0.02, info_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='bottom', horizontalalignment='right',
           bbox=dict(boxstyle="round,pad=0.4", facecolor="lightgray", alpha=0.8))
    
    # Ajusta limites dos eixos
    ax.set_xlim(-0.02, df['I_A'].max() * 1.05)
    ax.set_ylim(-0.1, df['B_mT'].max() * 1.05)
    
    # Formatação dos ticks
    ax.tick_params(axis='both', which='major', labelsize=11)
    
    # Layout final
    plt.tight_layout()
    
    # Salva gráfico
    output_file = output_path / "sonda_hall_campo_magnetico.png"
    fig.savefig(output_file, dpi=300, bbox_inches='tight', 
               facecolor='white', edgecolor='none')
    
    print(f"\nGráfico salvo em: {output_file}")
    
    # Fecha figura para liberar memória
    plt.close(fig)
    
    return output_file

def print_summary_statistics(df, B_theoretical_mT, K_experimental):
    """
    Imprime estatísticas resumo da análise.
    
    Args:
        df: DataFrame com dados experimentais  
        B_theoretical_mT: valores teóricos calculados
        K_experimental: coeficiente angular experimental
    """
    print(f"\n{'='*60}")
    print(f"RESUMO ESTATÍSTICO - EXPERIMENTO 5-EPR")
    print(f"{'='*60}")
    
    print(f"\nDados experimentais:")
    print(f"  Número de pontos: {len(df)}")
    print(f"  Faixa de corrente: {df['I_A'].min():.2f} - {df['I_A'].max():.2f} A")
    print(f"  Faixa de campo: {df['B_mT'].min():.2f} - {df['B_mT'].max():.2f} mT")
    
    print(f"\nCoeficientes angulares (T/A):")
    print(f"  Teórico: {K_theoretical:.4e}")
    print(f"  Experimental: {K_experimental:.4e}")
    print(f"  Concordância: {(K_experimental/K_theoretical)*100:.1f}%")
    
    # Calcula R² para avaliar qualidade do ajuste linear
    B_exp_T = df['B_T'].values
    B_pred_T = K_experimental * df['I_A'].values
    
    ss_res = np.sum((B_exp_T - B_pred_T) ** 2)
    ss_tot = np.sum((B_exp_T - np.mean(B_exp_T)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    
    print(f"\nQualidade do ajuste linear:")
    print(f"  R² = {r_squared:.6f}")
    print(f"  Linearidade: {'Excelente' if r_squared > 0.999 else 'Boa' if r_squared > 0.99 else 'Regular'}")
    
    print(f"\n{'='*60}")

def main():
    """
    Função principal - executa análise completa da sonda Hall.
    """
    print(f"Iniciando análise da Sonda Hall - Experimento 5-EPR...\n")
    
    # Verifica se arquivo de dados existe
    if not Path("Data/sonda_hall.csv").exists():
        print("Erro: Arquivo Data/sonda_hall.csv não encontrado!")
        return
    
    # Carrega dados experimentais
    df = load_experimental_data()
    
    # Analisa concordância teoria vs experimento
    B_theoretical_mT, K_experimental = analyze_data(df)
    
    # Gera gráfico
    print(f"\nGerando gráfico...")
    output_file = create_hall_plot(df)
    
    # Mostra estatísticas resumo
    print_summary_statistics(df, B_theoretical_mT, K_experimental)
    
    print(f"\nAnálise concluída!")
    print(f"- {len(df)} pontos experimentais analisados")
    print(f"- Curva teórica B = μ₀(4/5)^(3/2) × n/r × I calculada")
    print(f"- Gráfico scatter plot + curva teórica gerado")
    print(f"- Concordância teoria-experimento avaliada")
    
    return df, K_experimental

if __name__ == "__main__":
    result = main()
