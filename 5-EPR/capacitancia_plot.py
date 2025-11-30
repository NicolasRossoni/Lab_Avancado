#!/usr/bin/env python3
"""
capacitancia_plot.py

Plota dados de tensão vs frequência para diferentes capacitâncias.

Dados:
- 31 pF, 38 pF, 75 pF: bobina ativa e passiva
- Referência: apenas bobina ativa
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# =========================
# CONFIGURAÇÕES
# =========================

DATA_DIR = Path("Data")
OUTPUT_DIR = Path("Graficos")

# Arquivos de dados
FILES = {
    "31pF": "Lab Avançado 2025.2 - 31pF.csv",
    "38pF": "Lab Avançado 2025.2 - 38pF.csv",
    "75pF": "Lab Avançado 2025.2 - 75pF.csv",
    "ref": "Lab Avançado 2025.2 - ref.csv"
}

# Cores para cada capacitância
COLORS = {
    "31pF": "#E63946",    # Vermelho
    "38pF": "#2E86AB",    # Azul
    "75pF": "#06A77D",    # Verde
    "ref": "#000000"      # Preto
}

print("="*70)
print("PLOTAGEM DE DADOS - CAPACITÂNCIA VS FREQUÊNCIA")
print("="*70)


def load_capacitance_data(filename):
    """
    Carrega dados de capacitância do arquivo CSV.

    Args:
        filename: nome do arquivo

    Returns:
        DataFrame com dados
    """
    filepath = DATA_DIR / filename

    if not filepath.exists():
        print(f"AVISO: Arquivo {filepath} não encontrado. Pulando...")
        return None

    # Lê CSV com vírgula como separador decimal (formato brasileiro)
    df = pd.read_csv(filepath, decimal=',')

    print(f"\nArquivo: {filename}")
    print(f"  {len(df)} pontos")
    print(f"  Colunas: {', '.join(df.columns)}")

    return df


def create_combined_plot(all_data, output_dir="Graficos"):
    """
    Cria gráfico combinado com todos os dados.

    Args:
        all_data: dicionário com DataFrames de cada arquivo
        output_dir: diretório para salvar o gráfico
    """
    # Cria diretório se não existir
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Configura figura
    fig, ax = plt.subplots(figsize=(12, 9))

    # Plota dados de cada capacitância
    for label, df in all_data.items():
        if df is None:
            continue

        color = COLORS[label]

        # Para arquivos com capacitância (têm bobina ativa e passiva)
        if label != "ref":
            # Bobina passiva
            ax.plot(df['ν (MHz)'], df['V_passiva (V)'],
                   'o-', color=color, markersize=6, linewidth=2, alpha=0.8,
                   label=f'{label} (Passiva)', zorder=2)

            # Bobina ativa
            ax.plot(df['ν (MHz)'], df['V_ativa (V)'],
                   's-', color=color, markersize=6, linewidth=2, alpha=0.8,
                   label=f'{label} (Ativa)', zorder=2)
        else:
            # Referência (apenas bobina ativa)
            ax.plot(df['ν (MHz)'], df['V_ativa (V)'],
                   'o-', color=color, markersize=7, linewidth=2.5, alpha=0.9,
                   label='ref', zorder=3)

    # Formatação
    ax.set_xlabel('Frequência (MHz)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Tensão (V)', fontsize=14, fontweight='bold')
    ax.set_title('EPR: Tensão vs Frequência para Diferentes Capacitâncias',
                fontsize=16, fontweight='bold', pad=20)

    # Grid
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.8)
    ax.set_axisbelow(True)

    # Legenda
    legend = ax.legend(fontsize=11, loc='lower right',
                      frameon=True, fancybox=True, shadow=True,
                      ncol=2)
    legend.get_frame().set_alpha(0.9)

    # Formatação dos ticks
    ax.tick_params(axis='both', which='major', labelsize=11)

    # Layout
    plt.tight_layout()

    # Salva gráfico
    output_file = output_path / "capacitancia_frequencia.png"
    fig.savefig(output_file, dpi=300, bbox_inches='tight',
               facecolor='white', edgecolor='none')

    print(f"\n{'='*70}")
    print(f"Gráfico salvo: {output_file}")
    print(f"{'='*70}")

    plt.close(fig)

    return output_file


def main():
    """
    Função principal - carrega e plota todos os dados.
    """
    print("\nCarregando dados...\n")

    # Carrega todos os arquivos
    all_data = {}
    for label, filename in FILES.items():
        df = load_capacitance_data(filename)
        all_data[label] = df

    # Cria gráfico combinado
    print("\nGerando gráfico combinado...")
    create_combined_plot(all_data, OUTPUT_DIR)

    print("\nAnálise concluída!")

    return all_data


if __name__ == "__main__":
    result = main()
