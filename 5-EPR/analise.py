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
    Calcula a largura à meia altura (FWHM) de um pico invertido (gaussiana negativa).

    Args:
        x: array com eixo X (tempo)
        y: array com eixo Y (voltagem)
        label: rótulo para mensagens de debug

    Returns:
        dict: {
            'baseline': valor da baseline (primeiro ponto),
            'min_value': valor mínimo (pico),
            'min_index': índice do mínimo,
            'peak_height': altura do pico,
            'half_height': valor da meia-altura,
            'left_x': coordenada X esquerda na meia-altura,
            'right_x': coordenada X direita na meia-altura,
            'fwhm': largura à meia altura
        }
    """
    print(f"\n{'─'*70}")
    print(f"CÁLCULO FWHM - {label}")
    print(f"{'─'*70}")

    # Baseline = primeiro ponto (após nivelamento)
    baseline = y[0]

    # Encontra o ponto mínimo (pico negativo)
    min_index = np.argmin(y)
    min_value = y[min_index]
    min_x = x[min_index]

    # Altura do pico (positiva, medida da baseline até o mínimo)
    peak_height = baseline - min_value

    # Meia-altura (valor de Y na meia-altura)
    half_height = baseline - (peak_height / 2.0)

    print(f"Baseline (primeiro ponto): {baseline:.4f} V")
    print(f"Pico mínimo: {min_value:.4f} V em t = {min_x*1000:.4f} ms")
    print(f"Altura do pico: {peak_height:.4f} V")
    print(f"Meia-altura: {half_height:.4f} V")

    # Encontra os pontos onde a curva cruza a meia-altura
    # Procura à esquerda e à direita do pico

    # Lado esquerdo: do início até o pico
    left_indices = np.where((x < min_x) & (y <= half_height))[0]
    if len(left_indices) == 0:
        print("  AVISO: Não encontrou cruzamento à esquerda, usando interpolação")
        # Interpola para encontrar o ponto exato
        left_region_x = x[:min_index]
        left_region_y = y[:min_index]
        if len(left_region_x) > 1:
            f_left = interp1d(left_region_y[::-1], left_region_x[::-1],
                            kind='linear', fill_value='extrapolate')
            left_x = float(f_left(half_height))
        else:
            left_x = x[0]
    else:
        # Usa o último ponto antes do cruzamento e interpola
        idx = left_indices[-1]
        if idx + 1 < min_index:
            # Interpolação linear entre idx e idx+1
            x1, y1 = x[idx], y[idx]
            x2, y2 = x[idx + 1], y[idx + 1]
            if y2 != y1:
                left_x = x1 + (half_height - y1) * (x2 - x1) / (y2 - y1)
            else:
                left_x = x1
        else:
            left_x = x[idx]

    # Lado direito: do pico até o final
    right_indices = np.where((x > min_x) & (y <= half_height))[0]
    if len(right_indices) == 0:
        print("  AVISO: Não encontrou cruzamento à direita, usando interpolação")
        # Interpola para encontrar o ponto exato
        right_region_x = x[min_index:]
        right_region_y = y[min_index:]
        if len(right_region_x) > 1:
            f_right = interp1d(right_region_y, right_region_x,
                             kind='linear', fill_value='extrapolate')
            right_x = float(f_right(half_height))
        else:
            right_x = x[-1]
    else:
        # Usa o primeiro ponto após o cruzamento e interpola
        idx = right_indices[0]
        if idx > min_index:
            # Interpolação linear entre idx-1 e idx
            x1, y1 = x[idx - 1], y[idx - 1]
            x2, y2 = x[idx], y[idx]
            if y2 != y1:
                right_x = x1 + (half_height - y1) * (x2 - x1) / (y2 - y1)
            else:
                right_x = x2
        else:
            right_x = x[idx]

    # Largura à meia altura (FWHM)
    fwhm = right_x - left_x

    print(f"Ponto esquerdo (meia-altura): t = {left_x*1000:.4f} ms")
    print(f"Ponto direito (meia-altura):  t = {right_x*1000:.4f} ms")
    print(f"FWHM: {fwhm*1000:.4f} ms ({fwhm*1e6:.2f} μs)")

    return {
        'baseline': baseline,
        'min_value': min_value,
        'min_index': min_index,
        'min_x': min_x,
        'peak_height': peak_height,
        'half_height': half_height,
        'left_x': left_x,
        'right_x': right_x,
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

    # Plota linhas de meia-altura (FWHM) - 300 K
    fwhm_amb_added_to_legend = False
    if not fwhm_amb_added_to_legend:
        ax.plot([fwhm_ambiente['left_x']*1000, fwhm_ambiente['right_x']*1000],
               [fwhm_ambiente['half_height'], fwhm_ambiente['half_height']],
               color=color_fwhm, linewidth=2, linestyle='--', alpha=0.7,
               label='Meia-altura (FWHM)', zorder=2)
        fwhm_amb_added_to_legend = True
    else:
        ax.plot([fwhm_ambiente['left_x']*1000, fwhm_ambiente['right_x']*1000],
               [fwhm_ambiente['half_height'], fwhm_ambiente['half_height']],
               color=color_fwhm, linewidth=2, linestyle='--', alpha=0.7, zorder=2)

    # Marcadores verticais nos extremos - 300 K
    ax.plot([fwhm_ambiente['left_x']*1000, fwhm_ambiente['left_x']*1000],
           [fwhm_ambiente['baseline'], fwhm_ambiente['half_height']],
           color=color_fwhm, linewidth=1.5, linestyle=':', alpha=0.5, zorder=1)
    ax.plot([fwhm_ambiente['right_x']*1000, fwhm_ambiente['right_x']*1000],
           [fwhm_ambiente['baseline'], fwhm_ambiente['half_height']],
           color=color_fwhm, linewidth=1.5, linestyle=':', alpha=0.5, zorder=1)

    # Plota linhas de meia-altura (FWHM) - 77 K
    ax.plot([fwhm_nitrogenio['left_x']*1000, fwhm_nitrogenio['right_x']*1000],
           [fwhm_nitrogenio['half_height'], fwhm_nitrogenio['half_height']],
           color=color_fwhm, linewidth=2, linestyle='--', alpha=0.7, zorder=2)

    # Marcadores verticais nos extremos - 77 K
    ax.plot([fwhm_nitrogenio['left_x']*1000, fwhm_nitrogenio['left_x']*1000],
           [fwhm_nitrogenio['baseline'], fwhm_nitrogenio['half_height']],
           color=color_fwhm, linewidth=1.5, linestyle=':', alpha=0.5, zorder=1)
    ax.plot([fwhm_nitrogenio['right_x']*1000, fwhm_nitrogenio['right_x']*1000],
           [fwhm_nitrogenio['baseline'], fwhm_nitrogenio['half_height']],
           color=color_fwhm, linewidth=1.5, linestyle=':', alpha=0.5, zorder=1)

    # Formatação
    ax.set_xlabel('Tempo (ms)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Voltagem (V)', fontsize=14, fontweight='bold')
    ax.set_title('EPR: Largura à Meia-Altura (FWHM) em Diferentes Temperaturas',
                fontsize=16, fontweight='bold', pad=20)

    # Grid
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.8)
    ax.set_axisbelow(True)

    # Legenda
    legend = ax.legend(fontsize=12, loc='upper right',
                      frameon=True, fancybox=True, shadow=True)
    legend.get_frame().set_alpha(0.95)

    # Adiciona linha na baseline para referência
    ax.axhline(y=fwhm_ambiente['baseline'], color='gray',
              linestyle='--', linewidth=1, alpha=0.4, zorder=0)

    # Texto com valores de FWHM
    fwhm_text = (
        f"Largura à Meia-Altura (FWHM):\n"
        f"\n"
        f"300 K:  {fwhm_ambiente['fwhm']*1000:.4f} ms\n"
        f"        ({fwhm_ambiente['fwhm']*1e6:.2f} μs)\n"
        f"\n"
        f"77 K:   {fwhm_nitrogenio['fwhm']*1000:.4f} ms\n"
        f"        ({fwhm_nitrogenio['fwhm']*1e6:.2f} μs)\n"
        f"\n"
        f"Razão (77K/300K): {fwhm_nitrogenio['fwhm']/fwhm_ambiente['fwhm']:.2f}"
    )

    ax.text(0.02, 0.05, fwhm_text, transform=ax.transAxes,
           fontsize=11, verticalalignment='bottom', horizontalalignment='left',
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
    print(f"\n{'='*70}")
    print("RESUMO DA ANÁLISE")
    print(f"{'='*70}")
    print(f"\nLargura à Meia-Altura (FWHM):")
    print(f"  300 K: {fwhm_ambiente['fwhm']*1000:.4f} ms ({fwhm_ambiente['fwhm']*1e6:.2f} μs)")
    print(f"  77 K:  {fwhm_nitrogenio['fwhm']*1000:.4f} ms ({fwhm_nitrogenio['fwhm']*1e6:.2f} μs)")
    print(f"\nRazão FWHM (77K/300K): {fwhm_nitrogenio['fwhm']/fwhm_ambiente['fwhm']:.3f}")
    print(f"\nGráfico gerado: {output_file}")
    print(f"{'='*70}\n")

    return df, fwhm_ambiente, fwhm_nitrogenio


if __name__ == "__main__":
    result = main()
