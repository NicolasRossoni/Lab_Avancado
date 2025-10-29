#!/usr/bin/env python3
"""
Cálculo da constante de Boltzmann (k_B) usando curva I-V do LED vermelho8.

Relação teórica: I = I0 * exp(e*V/(n*k_B*T))
Aplicando ln: ln(I) = ln(I0) + (e/(n*k_B*T)) * V

Portanto: k_B = e/(n*S*T), onde S é o coeficiente angular do gráfico ln(I) vs V
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple

# =============================
# PARÂMETROS E CONSTANTES
# =============================
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "Dados" / "Tensao_Corrente"
OUTPUT_DIR = PROJECT_DIR / "Graficos"

# Constantes físicas
E_CHARGE = 1.602176634e-19  # Carga elementar em C
K_B_REF = 8.617333262e-5    # Constante de Boltzmann de referência em eV/K
T_KELVIN = 300.0             # Temperatura ambiente em K
N_IDEALIDADE = 1.0           # Fator de idealidade do diodo (assumido)

# =============================
# FUNÇÕES
# =============================

def load_vermelho8_data() -> Tuple[np.ndarray, np.ndarray]:
    """
    Carrega dados de tensão e corrente do vermelho8.
    Retorna (V, I) em (mV, mA).
    """
    file_path = DATA_DIR / "Lab Avançado 2025.2 - vermelho8.csv"
    
    if not file_path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {file_path}")
    
    data = np.loadtxt(file_path, delimiter=',', skiprows=1)
    V = data[:, 0]  # mV
    I = data[:, 1]  # mA
    
    return V, I

def linear_fit(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    """
    Fit linear y = a*x + b usando mínimos quadrados.
    Retorna (a, b, r²).
    """
    n = len(x)
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    
    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sum((x - x_mean) ** 2)
    
    a = numerator / denominator
    b = y_mean - a * x_mean
    
    # Calcula R²
    y_pred = a * x + b
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y_mean) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    
    return a, b, r_squared

def calculate_kb(S: float, T: float = T_KELVIN, n: float = N_IDEALIDADE) -> float:
    """
    Calcula k_B usando a fórmula: k_B = e/(n*S*T).
    
    Parâmetros:
    - S: coeficiente angular em 1/V (ln(mA) por V)
    - T: temperatura em K
    - n: fator de idealidade
    
    Retorna k_B em eV/K.
    """
    # S está em unidades de 1/mV, precisa converter para 1/V
    S_per_V = S * 1000.0  # converte de 1/mV para 1/V
    
    # k_B = e/(n*S*T), mas e está em C, então resultado é em J/K
    k_B_joule = E_CHARGE / (n * S_per_V * T)
    
    # Converte para eV/K
    k_B_eV = k_B_joule / E_CHARGE
    
    return k_B_eV

def plot_kb_analysis(V: np.ndarray, I: np.ndarray, output_dir: Path) -> None:
    """
    Gera gráfico ln(I) vs V com análise de k_B.
    """
    # Filtra apenas valores positivos de corrente
    mask = I > 0
    V_filtered = V[mask]
    I_filtered = I[mask]
    
    # Calcula ln(I)
    ln_I = np.log(I_filtered)
    
    # Fit linear
    S, b, r2 = linear_fit(V_filtered, ln_I)
    
    # Calcula k_B
    k_B_calc = calculate_kb(S*10)
    
    # Calcula desvio percentual
    delta_percent = abs(k_B_calc - K_B_REF) / K_B_REF * 100
    
    # Cria figura
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Plota dados
    ax.plot(V_filtered, ln_I, 'o', color='tab:red', markersize=6, 
            label='Dados experimentais', alpha=0.7)
    
    # Plota fit
    V_fit = np.linspace(V_filtered.min(), V_filtered.max(), 100)
    ln_I_fit = S * V_fit + b
    ax.plot(V_fit, ln_I_fit, '-', color='black', linewidth=2, 
            label='Ajuste linear')
    
    # Configurações do gráfico
    ax.set_xlabel('Tensão (mV)', fontsize=13)
    ax.set_ylabel('ln(I) [ln(mA)]', fontsize=13)
    ax.set_title('Cálculo de $k_B$ - Laser vermelho8', fontsize=14, weight='bold')
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend(loc='upper left', fontsize=11)
    
    # Texto com resultados (estilo similar ao graph4.py)
    textstr = (
        f'$S = {S:.4f}$ mV$^{{-1}}$\n'
        f'$R^2 = {r2:.6f}$\n'
        f'\n'
        f'$k_B = \\frac{{e}}{{n \\cdot S \\cdot T}}$\n'
        f'\n'
        f'$k_B = {k_B_calc:.4e}$ eV/K\n'
        f'$k_{{B,ref}} = {K_B_REF:.4e}$ eV/K\n'
        f'$\\Delta = {delta_percent:.2f}\\%$'
    )
    
    # Adiciona caixa de texto
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.62, 0.05, textstr, transform=ax.transAxes, fontsize=12,
            verticalalignment='bottom', bbox=props, family='monospace')
    
    # Salva figura
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "calculo_kb_vermelho8.png"
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"\n{'='*60}")
    print(f"ANÁLISE DA CONSTANTE DE BOLTZMANN (k_B)")
    print(f"{'='*60}")
    print(f"Laser: vermelho8")
    print(f"Temperatura: T = {T_KELVIN} K")
    print(f"Fator de idealidade: n = {N_IDEALIDADE}")
    print(f"\nCoeficiente angular: S = {S:.6f} mV⁻¹")
    print(f"                         S = {S*1000:.6f} V⁻¹")
    print(f"Coeficiente de determinação: R² = {r2:.6f}")
    print(f"\nConstante de Boltzmann calculada:")
    print(f"  k_B = {k_B_calc:.6e} eV/K")
    print(f"\nConstante de Boltzmann de referência:")
    print(f"  k_B,ref = {K_B_REF:.6e} eV/K")
    print(f"\nDesvio:")
    print(f"  Δ = {delta_percent:.2f}%")
    print(f"\nGráfico salvo em: {output_path}")
    print(f"{'='*60}\n")

def main():
    """Função principal."""
    print("Carregando dados do vermelho8...")
    V, I = load_vermelho8_data()
    
    print(f"Dados carregados: {len(V)} pontos")
    print(f"Faixa de tensão: {V.min():.2f} - {V.max():.2f} mV")
    print(f"Faixa de corrente: {I.min():.4f} - {I.max():.4f} mA")
    
    print("\nGerando análise de k_B...")
    plot_kb_analysis(V, I, OUTPUT_DIR)

if __name__ == "__main__":
    main()
