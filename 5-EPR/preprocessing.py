#!/usr/bin/env python3
"""
preprocessing.py

Pré-processamento de dados de EPR do osciloscópio.

Extrai dados do canal 2 (CH2) das medições em temperatura ambiente
e em nitrogênio líquido, gerando um arquivo CSV processado.

Estrutura dos arquivos do osciloscópio:
- Linhas 1-18: Metadados (Record Length, Sample Interval, etc.)
- Linha 19 em diante: Dados (tempo, voltagem)
- Formato: ,,,tempo,voltagem
"""

import pandas as pd
import numpy as np
from pathlib import Path

# =========================
# CONFIGURAÇÕES
# =========================

# Diretórios
DATA_DIR = Path("Data")
AMBIENTE_DIR = DATA_DIR / "ambiente"
NITROGENIO_DIR = DATA_DIR / "nitrogenio"

# Arquivos de entrada (Canal 2 apenas)
AMBIENTE_CH2 = "F0000CH2.CSV"
NITROGENIO_CH2 = "F0001CH2.CSV"

# Arquivo de saída
OUTPUT_FILE = DATA_DIR / "processed.csv"

# Linha onde começam os dados (1-indexed, conforme especificado)
DATA_START_LINE = 19

print("="*70)
print("PRÉ-PROCESSAMENTO DE DADOS EPR - OSCILOSCÓPIO")
print("="*70)


def read_oscilloscope_ch2(filepath):
    """
    Lê arquivo CSV do canal 2 do osciloscópio.

    O formato do arquivo é:
    - Linhas 1-18: Metadados
    - Linha 19+: Dados no formato: ,,,tempo,voltagem

    Args:
        filepath: caminho para o arquivo CH2.CSV

    Returns:
        DataFrame com colunas 'time' e 'voltage'
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {filepath}")

    print(f"\nLendo: {filepath.name}")
    print(f"  Caminho: {filepath.parent.name}/")

    # Lê arquivo pulando as primeiras 18 linhas (metadados)
    # skiprows=18 pula linhas 0-17, começando a ler da linha 18 (índice 0-based)
    # que corresponde à linha 19 (índice 1-based)
    df = pd.read_csv(filepath, skiprows=18, header=None)

    # Remove as três primeiras colunas vazias (colunas 0, 1, 2)
    # Fica com colunas 3 e 4 que contêm tempo e voltagem
    df = df.iloc[:, 3:5]

    # Renomeia colunas
    df.columns = ['time', 'voltage']

    # Remove espaços em branco das strings (se houver)
    df['time'] = df['time'].astype(str).str.strip()
    df['voltage'] = df['voltage'].astype(str).str.strip()

    # Converte para numérico
    df['time'] = pd.to_numeric(df['time'], errors='coerce')
    df['voltage'] = pd.to_numeric(df['voltage'], errors='coerce')

    # Remove linhas com valores NaN (se houver)
    initial_len = len(df)
    df = df.dropna()
    final_len = len(df)

    if initial_len != final_len:
        print(f"  AVISO: {initial_len - final_len} linhas removidas (valores inválidos)")

    print(f"  Dados lidos: {len(df)} pontos")
    print(f"  Tempo: {df['time'].min():.6f} - {df['time'].max():.6f} s")
    print(f"  Voltagem: {df['voltage'].min():.3f} - {df['voltage'].max():.3f} V")

    return df


def verify_time_axes_match(df_ambiente, df_nitrogenio, tolerance=1e-9):
    """
    Verifica se os eixos de tempo das duas medições são compatíveis.

    Args:
        df_ambiente: DataFrame com dados de ambiente
        df_nitrogenio: DataFrame com dados de nitrogênio
        tolerance: tolerância para diferença entre tempos

    Returns:
        bool: True se os eixos são compatíveis
    """
    print(f"\n{'─'*70}")
    print("VERIFICAÇÃO DE COMPATIBILIDADE DOS EIXOS DE TEMPO")
    print(f"{'─'*70}")

    # Verifica se têm mesmo número de pontos
    len_amb = len(df_ambiente)
    len_nit = len(df_nitrogenio)

    print(f"Número de pontos:")
    print(f"  Ambiente:   {len_amb}")
    print(f"  Nitrogênio: {len_nit}")

    if len_amb != len_nit:
        print(f"  AVISO: Número de pontos diferente!")
        min_len = min(len_amb, len_nit)
        print(f"  Usando apenas os primeiros {min_len} pontos de cada")
        return False, min_len

    # Verifica se os valores de tempo são iguais (dentro da tolerância)
    max_diff = np.max(np.abs(df_ambiente['time'].values - df_nitrogenio['time'].values))

    print(f"\nDiferença máxima entre eixos de tempo: {max_diff:.2e} s")

    if max_diff < tolerance:
        print(f"  ✓ Eixos de tempo são idênticos (diferença < {tolerance:.0e} s)")
        return True, len_amb
    else:
        print(f"  AVISO: Eixos de tempo têm pequenas diferenças")
        print(f"  Usando eixo de tempo da medição em ambiente")
        return True, len_amb


def level_curves(df_ambiente, df_nitrogenio):
    """
    Nivela as curvas usando o primeiro ponto como referência.

    Subtrai o offset do nitrogênio para que ambas as curvas comecem
    no mesmo nível (primeiro ponto de ambiente).

    Args:
        df_ambiente: DataFrame com dados de ambiente
        df_nitrogenio: DataFrame com dados de nitrogênio

    Returns:
        tuple: (df_ambiente, df_nitrogenio_leveled)
    """
    print(f"\n{'─'*70}")
    print("NIVELAMENTO DAS CURVAS")
    print(f"{'─'*70}")

    # Primeiro ponto de cada curva
    baseline_ambiente = df_ambiente['voltage'].iloc[0]
    baseline_nitrogenio = df_nitrogenio['voltage'].iloc[0]

    # Calcula offset
    offset = baseline_nitrogenio - baseline_ambiente

    print(f"Primeiro ponto (baseline):")
    print(f"  Ambiente:   {baseline_ambiente:.4f} V")
    print(f"  Nitrogênio: {baseline_nitrogenio:.4f} V")
    print(f"  Offset:     {offset:.4f} V")

    # Aplica correção: subtrai offset de todos os pontos de nitrogênio
    df_nitrogenio_leveled = df_nitrogenio.copy()
    df_nitrogenio_leveled['voltage'] = df_nitrogenio['voltage'] - offset

    # Verifica nivelamento
    new_baseline_nitrogenio = df_nitrogenio_leveled['voltage'].iloc[0]
    print(f"\nApós nivelamento:")
    print(f"  Ambiente:   {baseline_ambiente:.4f} V")
    print(f"  Nitrogênio: {new_baseline_nitrogenio:.4f} V")
    print(f"  ✓ Curvas niveladas!")

    return df_ambiente, df_nitrogenio_leveled


def create_processed_csv(df_ambiente, df_nitrogenio, output_file):
    """
    Cria arquivo CSV processado com três colunas: X, Ambiente, Nitrogenio.

    Args:
        df_ambiente: DataFrame com dados de ambiente
        df_nitrogenio: DataFrame com dados de nitrogênio
        output_file: caminho para arquivo de saída
    """
    print(f"\n{'─'*70}")
    print("CRIANDO ARQUIVO PROCESSADO")
    print(f"{'─'*70}")

    # Verifica compatibilidade dos eixos de tempo
    axes_match, n_points = verify_time_axes_match(df_ambiente, df_nitrogenio)

    # Trunca dataframes se necessário
    df_ambiente = df_ambiente.head(n_points).copy()
    df_nitrogenio = df_nitrogenio.head(n_points).copy()

    # Nivela as curvas
    df_ambiente, df_nitrogenio = level_curves(df_ambiente, df_nitrogenio)

    # Cria DataFrame processado
    df_processed = pd.DataFrame({
        'X': df_ambiente['time'].values,
        'Ambiente': df_ambiente['voltage'].values,
        'Nitrogenio': df_nitrogenio['voltage'].values
    })

    # Salva arquivo CSV
    output_path = Path(output_file)
    df_processed.to_csv(output_path, index=False)

    print(f"\nArquivo criado: {output_path}")
    print(f"  {len(df_processed)} pontos salvos")
    print(f"  Colunas: X, Ambiente, Nitrogenio")

    # Mostra preview dos dados
    print(f"\nPreview dos dados:")
    print(df_processed.head(10).to_string(index=False))
    print("...")
    print(df_processed.tail(5).to_string(index=False))

    return df_processed


def main():
    """
    Função principal - executa pré-processamento completo.
    """
    print("\nIniciando pré-processamento...\n")

    # Verifica se diretórios existem
    if not AMBIENTE_DIR.exists():
        raise FileNotFoundError(f"Diretório não encontrado: {AMBIENTE_DIR}")

    if not NITROGENIO_DIR.exists():
        raise FileNotFoundError(f"Diretório não encontrado: {NITROGENIO_DIR}")

    # Lê dados do canal 2 - Ambiente
    ambiente_file = AMBIENTE_DIR / AMBIENTE_CH2
    df_ambiente = read_oscilloscope_ch2(ambiente_file)

    # Lê dados do canal 2 - Nitrogênio
    nitrogenio_file = NITROGENIO_DIR / NITROGENIO_CH2
    df_nitrogenio = read_oscilloscope_ch2(nitrogenio_file)

    # Cria arquivo processado
    df_processed = create_processed_csv(df_ambiente, df_nitrogenio, OUTPUT_FILE)

    # Resumo final
    print(f"\n{'='*70}")
    print("PRÉ-PROCESSAMENTO CONCLUÍDO")
    print(f"{'='*70}")
    print(f"\nArquivo gerado: {OUTPUT_FILE}")
    print(f"Dados prontos para análise!")
    print(f"{'='*70}\n")

    return df_processed


if __name__ == "__main__":
    result = main()
