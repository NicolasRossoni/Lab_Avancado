#!/usr/bin/env python3
"""
analise_extra.py

Análise dos dados medidos à mão (extra_by_hand.csv).
Calcula ΔB₀ e gera tabela com os resultados.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# =========================
# CONSTANTES
# =========================

# Parâmetros da bobina de Helmholtz
mu_0 = 4 * np.pi * 1e-7  # Permeabilidade do vácuo [Vs/Am]
n = 320  # Número de espiras
r = 6.8e-2  # Raio da bobina em metros (6.8 cm)
geometric_factor = (4/5)**(3/2)  # Fator geométrico Helmholtz

# Constante de conversão corrente -> campo magnético [T/A]
K_helmholtz = mu_0 * geometric_factor * n / r

# Incertezas experimentais
n_uncertainty = 1  # espiras
r_uncertainty = 0.001  # m (0.1 cm)
I_uncertainty = 0.01  # A (último algarismo significativo)

# Valor de literatura para DPPH
delta_B_dpph_min = 0.15  # mT
delta_B_dpph_max = 0.81  # mT

print("="*70)
print("ANÁLISE DE DADOS MEDIDOS À MÃO - ΔB₀")
print("="*70)
print(f"\nConstante K = μ₀ × (4/5)^(3/2) × n/r = {K_helmholtz:.4e} T/A")
print(f"K = {K_helmholtz*1000:.4f} mT/A")


def load_data():
    """Carrega dados do arquivo extra_by_hand.csv."""
    filepath = Path("Data/extra_by_hand.csv")

    # Lê CSV com vírgula como separador decimal
    df = pd.read_csv(filepath, decimal=',')

    print(f"\nDados carregados:")
    print(f"  Frequência: {df['frequencia'].iloc[0]:.1f} MHz")
    print(f"  I_pico: {df['pico'].iloc[0]:.2f} A")
    print(f"  I_meia_direita: {df['meia altura da direita'].iloc[0]:.2f} A")
    print(f"  I_meia_esquerda: {df['meia altura da esquerda'].iloc[0]:.2f} A")
    print(f"  ΔI: {df['delta'].iloc[0]:.2f} A")

    return df


def calculate_delta_B(delta_I):
    """
    Calcula ΔB₀ = ΔI × K com propagação de erro.

    Args:
        delta_I: variação de corrente [A]

    Returns:
        tuple: (ΔB₀ em mT, incerteza de ΔB₀ em mT)
    """
    print(f"\n{'─'*70}")
    print("CÁLCULO DE ΔB₀")
    print(f"{'─'*70}")

    # ΔB₀ = K × ΔI
    delta_B0_T = K_helmholtz * delta_I
    delta_B0_mT = delta_B0_T * 1000  # T -> mT

    print(f"\nΔB₀ = K × ΔI")
    print(f"ΔB₀ = {K_helmholtz:.4e} T/A × {delta_I:.4f} A")
    print(f"ΔB₀ = {delta_B0_T:.4e} T = {delta_B0_mT:.4f} mT")

    # Propagação de erro
    # Para ΔI medido: σ_ΔI = √(σ_I_esq² + σ_I_dir²)
    sigma_delta_I = np.sqrt(2 * I_uncertainty**2)

    print(f"\nIncerteza de ΔI:")
    print(f"  σ_ΔI = √(σ_I² + σ_I²) = √(2 × {I_uncertainty}²)")
    print(f"  σ_ΔI = {sigma_delta_I:.4f} A")

    # Para K: σ_K/K = √[(σ_n/n)² + (σ_r/r)²]
    relative_n = n_uncertainty / n
    relative_r = r_uncertainty / r
    relative_K = np.sqrt(relative_n**2 + relative_r**2)

    print(f"\nIncerteza relativa de K:")
    print(f"  σ_K/K = √[(σ_n/n)² + (σ_r/r)²]")
    print(f"  σ_K/K = √[({n_uncertainty}/{n})² + ({r_uncertainty}/{r})²]")
    print(f"  σ_K/K = {relative_K:.6f}")

    # Para ΔB₀: σ_ΔB₀/ΔB₀ = √[(σ_K/K)² + (σ_ΔI/ΔI)²]
    relative_delta_I = sigma_delta_I / delta_I
    relative_delta_B0 = np.sqrt(relative_K**2 + relative_delta_I**2)

    sigma_delta_B0_mT = delta_B0_mT * relative_delta_B0

    print(f"\nIncerteza de ΔB₀:")
    print(f"  σ_ΔB₀/ΔB₀ = √[(σ_K/K)² + (σ_ΔI/ΔI)²]")
    print(f"  σ_ΔB₀/ΔB₀ = √[{relative_K:.6f}² + ({sigma_delta_I:.4f}/{delta_I:.4f})²]")
    print(f"  σ_ΔB₀/ΔB₀ = {relative_delta_B0:.6f}")
    print(f"  σ_ΔB₀ = {sigma_delta_B0_mT:.4f} mT")

    return delta_B0_mT, sigma_delta_B0_mT


def format_uncertainty(value, uncertainty):
    """Formata valor e incerteza com algarismos significativos."""
    if uncertainty == 0:
        return f"{value:.4f}", "0.0000"

    # Encontra ordem de magnitude da incerteza
    magnitude = 10 ** np.floor(np.log10(uncertainty))

    # Arredonda incerteza para 1 alg. sig.
    uncertainty_rounded = np.round(uncertainty / magnitude) * magnitude

    # Determina número de casas decimais
    if magnitude >= 1:
        decimals = 0
    else:
        decimals = int(-np.floor(np.log10(magnitude)))

    # Arredonda valor para mesma precisão
    value_rounded = np.round(value / magnitude) * magnitude

    # Formata strings
    if decimals == 0:
        value_str = f"{int(value_rounded)}"
        unc_str = f"{int(uncertainty_rounded)}"
    else:
        value_str = f"{value_rounded:.{decimals}f}"
        unc_str = f"{uncertainty_rounded:.{decimals}f}"

    return value_str, unc_str


def create_table(df, delta_B0_mT, sigma_delta_B0_mT, output_dir="Graficos"):
    """Cria tabela com dados e cálculo de ΔB₀."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Extrai dados
    freq = df['frequencia'].iloc[0]
    I_pico = df['pico'].iloc[0]
    I_dir = df['meia altura da direita'].iloc[0]
    I_esq = df['meia altura da esquerda'].iloc[0]
    delta_I = df['delta'].iloc[0]

    # Prepara dados da tabela
    table_data = [[
        f"{freq:.1f}",
        f"{I_pico:.2f}",
        f"{I_dir:.2f}",
        f"{I_esq:.2f}",
        f"{delta_I:.2f}"
    ]]

    # Cria figura (aumentada em Y)
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('tight')
    ax.axis('off')

    # Título
    title_text = "Dados Medidos à Mão - Análise EPR"
    ax.set_title(title_text, fontsize=16, weight='bold', pad=30)

    # Cabeçalhos com símbolos gregos
    headers = [
        "ν (MHz)",
        "I$_{pico}$ (A)",
        "I$_{meia\\_dir}$ (A)",
        "I$_{meia\\_esq}$ (A)",
        "$\\delta$I (A)"
    ]

    # Cria tabela
    table = ax.table(
        cellText=table_data,
        colLabels=headers,
        cellLoc='center',
        loc='center',
        colWidths=[0.15, 0.18, 0.22, 0.22, 0.15]
    )

    # Formatação da tabela
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)

    # Estiliza cabeçalho (azul)
    for i in range(5):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white', fontsize=12)

    # Formata ΔB₀ com incerteza
    delta_B0_str, sigma_B0_str = format_uncertainty(delta_B0_mT, sigma_delta_B0_mT)

    # Texto com cálculo de δB₀ (em negrito)
    calc_text = (
        f"$\\delta$B$_0$ = $\\delta$I × K = $\\delta$I × [$\\mu_0$ × (4/5)$^{{3/2}}$ × n/r]\n"
        f"\n"
        f"$\\delta$B$_0$ = {delta_B0_str} ± {sigma_B0_str} mT"
    )

    ax.text(0.5, 0.32, calc_text, ha='center', fontsize=13,
            weight='bold', color='black', transform=ax.transAxes)

    # Texto com valor de literatura (em itálico)
    lit_text = (
        f"$\\delta$B$_0$(DPPH) = {delta_B_dpph_min:.2f}−{delta_B_dpph_max:.2f} mT\n"
        f"Literatura: Intervalo de campo magnético para DPPH"
    )

    ax.text(0.5, 0.15, lit_text, ha='center', fontsize=11,
            style='italic', color='#555555', transform=ax.transAxes)

    # Salva tabela
    output_file = output_path / "tabela_delta_B0.png"
    fig.tight_layout()
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"\n{'='*70}")
    print(f"Tabela salva: {output_file}")
    print(f"{'='*70}")

    return output_file


def main():
    """Função principal."""
    # Carrega dados
    df = load_data()

    # Extrai ΔI
    delta_I = df['delta'].iloc[0]

    # Calcula ΔB₀
    delta_B0_mT, sigma_delta_B0_mT = calculate_delta_B(delta_I)

    # Formata resultado
    delta_B0_str, sigma_B0_str = format_uncertainty(delta_B0_mT, sigma_delta_B0_mT)

    print(f"\n{'='*70}")
    print("RESULTADO FINAL")
    print(f"{'='*70}")
    print(f"\nΔB₀ = {delta_B0_str} ± {sigma_B0_str} mT")
    print(f"\nValor de literatura: ΔB₀(DPPH) = {delta_B_dpph_min:.2f}−{delta_B_dpph_max:.2f} mT")

    # Verifica se está dentro do intervalo
    if delta_B_dpph_min <= delta_B0_mT <= delta_B_dpph_max:
        print(f"✓ Valor experimental está DENTRO do intervalo da literatura!")
    else:
        print(f"✗ Valor experimental está FORA do intervalo da literatura.")

    print(f"{'='*70}")

    # Gera tabela
    print(f"\nGerando tabela...")
    create_table(df, delta_B0_mT, sigma_delta_B0_mT)

    print(f"\nAnálise concluída!")

    return df, delta_B0_mT, sigma_delta_B0_mT


if __name__ == "__main__":
    result = main()
