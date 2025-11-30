#!/usr/bin/env python3
"""
analise.py

Análise e visualização dos dados de EPR em diferentes temperaturas.

Plota os sinais de EPR medidos em:
- 300 K (temperatura ambiente)
- 77 K (nitrogênio líquido)

Calcula a largura à meia altura (FWHM - Full Width at Half Maximum)
para cada sinal de EPR.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import interp1d

# =========================
# CONFIGURAÇÕES
# =========================

# Arquivo de entrada
INPUT_FILE = Path("Data/processed.csv")

# Diretório de saída
OUTPUT_DIR = Path("Graficos")

print("="*70)
print("ANÁLISE EPR - LARGURA À MEIA ALTURA (FWHM)")
print("="*70)


def load_processed_data(filepath):
    """
    Carrega dados processados do arquivo CSV.

    Args:
        filepath: caminho para o arquivo processed.csv

    Returns:
        DataFrame com colunas X, Ambiente, Nitrogenio
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {filepath}")

    print(f"\nCarregando dados: {filepath}")

    df = pd.read_csv(filepath)

    print(f"  {len(df)} pontos carregados")
    print(f"  Colunas: {', '.join(df.columns)}")

    return df


def calculate_fwhm(x, y, label=""):
    """
    Calcula a largura à meia altura de uma gaussiana INVERTIDA (pico para baixo).

    Algoritmo:
    1. Encontra ponto de MÍNIMO (X_min, Y_min) - pico da gaussiana invertida
    2. Calcula meia altura = Y_min / 2 (ambos negativos)
    3. A partir do mínimo, anda para DIREITA até Y > meia_altura
    4. A partir do mínimo, anda para ESQUERDA até Y > meia_altura
    5. Largura = diferença em X entre os dois pontos

    Args:
        x: array com eixo X (tempo)
        y: array com eixo Y (voltagem, negativa)
        label: rótulo para mensagens de debug

    Returns:
        dict com informações da meia altura
    """
    print(f"\n{'─'*70}")
    print(f"CÁLCULO LARGURA À MEIA ALTURA - {label}")
    print(f"{'─'*70}")

    # 1. Encontra o ponto de MÍNIMO (pico da gaussiana invertida)
    min_index = np.argmin(y)
    min_value = y[min_index]
    min_x = x[min_index]

    # 2. Calcula meia altura = Y_min / 2 (ambos negativos, então meia altura é menos negativa)
    half_height = min_value / 2.0

    print(f"Ponto de mínimo (pico): Y = {min_value:.4f} V em t = {min_x*1000:.4f} ms")
    print(f"Meia altura: {half_height:.4f} V")

    # 3. A partir do mínimo, anda para a DIREITA até Y > meia_altura
    # (saindo do pico negativo em direção a valores menos negativos)
    right_x = None
    right_y = None
    for i in range(min_index, len(y)):
        if y[i] > half_height:
            right_x = x[i]
            right_y = y[i]
            print(f"Ponto DIREITO encontrado: t = {right_x*1000:.4f} ms, Y = {right_y:.4f} V (índice {i})")
            break

    if right_x is None:
        # Não encontrou, usa último ponto
        right_x = x[-1]
        right_y = y[-1]
        print(f"  AVISO: Não encontrou cruzamento à direita, usando último ponto")

    # 4. A partir do mínimo, anda para a ESQUERDA até Y > meia_altura
    left_x = None
    left_y = None
    for i in range(min_index, -1, -1):
        if y[i] > half_height:
            left_x = x[i]
            left_y = y[i]
            print(f"Ponto ESQUERDO encontrado: t = {left_x*1000:.4f} ms, Y = {left_y:.4f} V (índice {i})")
            break

    if left_x is None:
        # Não encontrou, usa primeiro ponto
        left_x = x[0]
        left_y = y[0]
        print(f"  AVISO: Não encontrou cruzamento à esquerda, usando primeiro ponto")

    # 5. Largura = diferença em X
    fwhm = right_x - left_x

    print(f"Largura à meia altura: {fwhm*1000:.4f} ms")

    return {
        'min_value': min_value,
        'min_index': min_index,
        'min_x': min_x,
        'half_height': half_height,
        'left_x': left_x,     # Ponto da ESQUERDA
        'left_y': left_y,
        'right_x': right_x,   # Ponto da DIREITA
        'right_y': right_y,
        'fwhm': fwhm
    }


def create_epr_fwhm_plot(df, fwhm_ambiente, fwhm_nitrogenio, output_dir="Graficos"):
    """
    Cria gráfico de EPR com visualização da largura à meia altura.

    Args:
        df: DataFrame com dados processados
        fwhm_ambiente: dict com resultados FWHM para ambiente
        fwhm_nitrogenio: dict com resultados FWHM para nitrogênio
        output_dir: diretório para salvar o gráfico
    """
    print(f"\n{'─'*70}")
    print("GERANDO GRÁFICO COM FWHM")
    print(f"{'─'*70}")

    # Cria diretório se não existir
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Converte tempo para milissegundos
    time_ms = df['X'].values * 1000

    # Configura figura
    fig, ax = plt.subplots(figsize=(14, 9))

    # Cores
    color_ambiente = '#E63946'      # Vermelho para 300 K
    color_nitrogenio = '#457B9D'    # Azul para 77 K
    color_fwhm = '#2A2A2A'          # Preto para linhas FWHM

    # Plota curvas
    ax.plot(time_ms, df['Ambiente'].values,
           color=color_ambiente, linewidth=2.5, alpha=0.9,
           label='300 K (Temperatura Ambiente)', zorder=3)

    ax.plot(time_ms, df['Nitrogenio'].values,
           color=color_nitrogenio, linewidth=2.5, alpha=0.9,
           label='77 K (Nitrogênio Líquido)', zorder=3)

    # Reta horizontal tracejada na meia-altura - 300 K
    ax.plot([fwhm_ambiente['right_x']*1000, fwhm_ambiente['left_x']*1000],
           [fwhm_ambiente['half_height'], fwhm_ambiente['half_height']],
           color=color_fwhm, linewidth=2, linestyle='--', alpha=0.7,
           label='Meia-altura', zorder=2)

    # Linhas verticais tracejadas nos extremos - 300 K
    ax.plot([fwhm_ambiente['right_x']*1000, fwhm_ambiente['right_x']*1000],
           [0, fwhm_ambiente['half_height']],
           color=color_fwhm, linewidth=1.5, linestyle=':', alpha=0.5, zorder=1)
    ax.plot([fwhm_ambiente['left_x']*1000, fwhm_ambiente['left_x']*1000],
           [0, fwhm_ambiente['half_height']],
           color=color_fwhm, linewidth=1.5, linestyle=':', alpha=0.5, zorder=1)

    # Reta horizontal tracejada na meia-altura - 77 K
    ax.plot([fwhm_nitrogenio['right_x']*1000, fwhm_nitrogenio['left_x']*1000],
           [fwhm_nitrogenio['half_height'], fwhm_nitrogenio['half_height']],
           color=color_fwhm, linewidth=2, linestyle='--', alpha=0.7, zorder=2)

    # Linhas verticais tracejadas nos extremos - 77 K
    ax.plot([fwhm_nitrogenio['right_x']*1000, fwhm_nitrogenio['right_x']*1000],
           [0, fwhm_nitrogenio['half_height']],
           color=color_fwhm, linewidth=1.5, linestyle=':', alpha=0.5, zorder=1)
    ax.plot([fwhm_nitrogenio['left_x']*1000, fwhm_nitrogenio['left_x']*1000],
           [0, fwhm_nitrogenio['half_height']],
           color=color_fwhm, linewidth=1.5, linestyle=':', alpha=0.5, zorder=1)

    # Formatação
    ax.set_xlabel('Tempo (ms)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Voltagem (V)', fontsize=14, fontweight='bold')
    ax.set_title('EPR: Largura à Meia Altura em Diferentes Temperaturas',
                fontsize=16, fontweight='bold', pad=20)

    # Grid
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.8)
    ax.set_axisbelow(True)

    # Legenda
    legend = ax.legend(fontsize=12, loc='upper right',
                      frameon=True, fancybox=True, shadow=True)
    legend.get_frame().set_alpha(0.95)

    # Calcula razão das meias alturas
    razao_larguras = fwhm_nitrogenio['fwhm'] / fwhm_ambiente['fwhm']

    # Texto com valores de largura
    largura_text = (
        f"Largura à Meia Altura:\n"
        f"\n"
        f"300 K:  {fwhm_ambiente['fwhm']*1000:.4f} ms\n"
        f"\n"
        f"77 K:   {fwhm_nitrogenio['fwhm']*1000:.4f} ms\n"
        f"\n"
        f"77K/300K = {77/300:.3f}\n"
        f"Razão das meias alturas = {razao_larguras:.3f}"
    )

    ax.text(0.02, 0.05, largura_text, transform=ax.transAxes,
           fontsize=15, verticalalignment='bottom', horizontalalignment='left',
           bbox=dict(boxstyle="round,pad=0.6", facecolor="lightgray", alpha=0.9),
           fontfamily='monospace')

    # Bloco de texto com estimativa de δB (inferior direito)
    delta_B_300 = 0.17  # mT (do experimento à mão)
    delta_B_77 = delta_B_300 * 0.948  # mT (ajustado pela razão)

    delta_B_text = (
        f"Valor estimado de δB:\n"
        f"\n"
        f"δB_300 = {delta_B_300:.2f} mT\n"
        f"\n"
        f"δB_77 = {delta_B_77:.2f} mT"
    )

    ax.text(0.98, 0.05, delta_B_text, transform=ax.transAxes,
           fontsize=17, verticalalignment='bottom', horizontalalignment='right',
           bbox=dict(boxstyle="round,pad=0.6", facecolor="lightgray", alpha=0.9),
           fontfamily='monospace')

    # Formatação dos ticks
    ax.tick_params(axis='both', which='major', labelsize=11)

    # Layout
    plt.tight_layout()

    # Salva gráfico
    output_file = output_path / "epr_fwhm_analysis.png"
    fig.savefig(output_file, dpi=300, bbox_inches='tight',
               facecolor='white', edgecolor='none')

    print(f"\nGráfico salvo: {output_file}")

    plt.close(fig)

    return output_file


def main():
    """
    Função principal - executa análise completa com FWHM.
    """
    print("\nIniciando análise...\n")

    # Verifica se arquivo de entrada existe
    if not INPUT_FILE.exists():
        print(f"ERRO: Arquivo {INPUT_FILE} não encontrado!")
        print("Execute primeiro o script preprocessing.py")
        return

    # Carrega dados processados
    df = load_processed_data(INPUT_FILE)

    # Calcula FWHM para temperatura ambiente (300 K)
    fwhm_ambiente = calculate_fwhm(
        df['X'].values,
        df['Ambiente'].values,
        label="300 K (Temperatura Ambiente)"
    )

    # Calcula FWHM para nitrogênio líquido (77 K)
    fwhm_nitrogenio = calculate_fwhm(
        df['X'].values,
        df['Nitrogenio'].values,
        label="77 K (Nitrogênio Líquido)"
    )

    # Gera gráfico com visualização FWHM
    output_file = create_epr_fwhm_plot(df, fwhm_ambiente, fwhm_nitrogenio, OUTPUT_DIR)

    # Resumo final
    razao_larguras = fwhm_nitrogenio['fwhm'] / fwhm_ambiente['fwhm']

    print(f"\n{'='*70}")
    print("RESUMO DA ANÁLISE")
    print(f"{'='*70}")
    print(f"\nLargura à Meia Altura:")
    print(f"  300 K: {fwhm_ambiente['fwhm']*1000:.4f} ms")
    print(f"  77 K:  {fwhm_nitrogenio['fwhm']*1000:.4f} ms")
    print(f"\nRazão (77K/300K): {razao_larguras:.3f}")
    print(f"Razão das meias alturas: {razao_larguras:.3f}")
    print(f"\nGráfico gerado: {output_file}")
    print(f"{'='*70}\n")

    return df, fwhm_ambiente, fwhm_nitrogenio


if __name__ == "__main__":
    result = main()
