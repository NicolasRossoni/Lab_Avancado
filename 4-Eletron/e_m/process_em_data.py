"""
process_em_data.py

Processa dados experimentais para determinação de e/m (razão carga/massa do elétron).

Experimento: Bobina de Helmholtz
- V_fixo.csv: Tensão fixa, variando corrente I e raio R
- I_fixo.csv: Corrente fixa, variando tensão V e raio R

Fórmula: e/m = 2 * V * ((1/(k*μ₀)) * (r/(N*I*R)))²
"""

import pandas as pd
import numpy as np
from pathlib import Path

# =========================
# CONSTANTES EXPERIMENTAIS
# =========================

# Bobina de Helmholtz
N = 154                      # número de espiras (exato)
r_bobina = 0.398             # raio da bobina em metros (39.8 cm)
u_r_bobina = 0.001           # incerteza do raio (0.1 cm)

# Constantes físicas
mu0 = 1.25663706212e-6       # permeabilidade do vácuo [N/A²]
u_mu0 = 0.0                  # tratada como exata

k = 0.716                    # fator geométrico Helmholtz
u_k = 0.0                    # tratada como exata

# Valor de referência de e/m
e_over_m_ref = 1.758820024e11  # C/kg (CODATA 2018)

def parse_csv_value(value_str):
    """
    Converte string com vírgula decimal para float.
    Exemplo: "2,004" -> 2.004
    """
    return float(value_str.replace(',', '.'))

def get_uncertainty_from_value(value):
    """
    Determina incerteza baseada no último algarismo significativo.
    Exemplo: 2.004 -> 0.001, 61.0 -> 0.1
    """
    value_str = str(value)
    
    if '.' in value_str:
        # Número de casas decimais
        decimals = len(value_str.split('.')[1])
        return 10 ** (-decimals)
    else:
        # Sem decimais, incerteza é 1
        return 1.0

def calculate_e_over_m(V, I, R_cm, r_bobina, N, k, mu0):
    """
    Calcula e/m usando a fórmula fornecida.
    
    Args:
        V: Tensão de aceleração [V]
        I: Corrente na bobina [A]
        R_cm: Raio da trajetória [cm]
        r_bobina: Raio da bobina [m]
        N: Número de espiras
        k: Fator geométrico
        mu0: Permeabilidade do vácuo [N/A²]
    
    Returns:
        float: e/m [C/kg]
    """
    # Converter R de cm para m
    R = R_cm / 100.0
    
    # Fórmula: e/m = 2 * V * ((1/(k*μ₀)) * (r/(N*I*R)))²
    term = (1.0 / (k * mu0)) * (r_bobina / (N * I * R))
    e_over_m = 2.0 * V * term**2
    
    return e_over_m

def calculate_uncertainty(V, u_V, I, u_I, R_cm, u_R_cm, r_bobina, u_r_bobina, 
                         N, u_N, k, u_k, mu0, u_mu0, e_over_m):
    """
    Calcula incerteza de e/m por propagação de erros.
    
    Fórmula: u_e/m = (e/m) * sqrt(
        (u_V/V)² + (2*u_r/r)² + (2*u_N/N)² + (2*u_I/I)² + 
        (2*u_R/R)² + (2*u_mu0/mu0)² + (2*u_k/k)²
    )
    """
    # Converter R de cm para m
    R = R_cm / 100.0
    u_R = u_R_cm / 100.0
    
    # Termos relativos
    term_V = (u_V / V)**2 if V != 0 else 0
    term_r = (2 * u_r_bobina / r_bobina)**2 if r_bobina != 0 else 0
    term_N = (2 * u_N / N)**2 if N != 0 and u_N != 0 else 0
    term_I = (2 * u_I / I)**2 if I != 0 else 0
    term_R = (2 * u_R / R)**2 if R != 0 else 0
    term_mu0 = (2 * u_mu0 / mu0)**2 if mu0 != 0 and u_mu0 != 0 else 0
    term_k = (2 * u_k / k)**2 if k != 0 and u_k != 0 else 0
    
    # Incerteza total
    u_e_over_m = e_over_m * np.sqrt(
        term_V + term_r + term_N + term_I + term_R + term_mu0 + term_k
    )
    
    return u_e_over_m

def process_V_fixo():
    """
    Processa dados de V_fixo.csv (tensão fixa, I e R variáveis).
    """
    print("="*70)
    print("PROCESSANDO: V_fixo.csv (Tensão Fixa)")
    print("="*70)
    
    # Carregar dados
    df = pd.read_csv("Data/V_fixo.csv")
    
    print(f"\n📂 Dados carregados: {len(df)} medidas")
    print(f"\nColunas: {list(df.columns)}")
    
    # Processar cada linha
    results = []
    
    for idx, row in df.iterrows():
        # Extrair valores (convertendo vírgula para ponto)
        I = parse_csv_value(row['I (A)'])
        V = parse_csv_value(row['Tensão fixa (V)'])
        R_cm = float(row['R (cm)'])
        
        # Determinar incertezas
        u_I = get_uncertainty_from_value(I)
        u_V = get_uncertainty_from_value(V)
        u_R_cm = 0.5  # Incerteza fixa de 0.5 cm para o raio
        
        # Calcular e/m
        e_over_m = calculate_e_over_m(V, I, R_cm, r_bobina, N, k, mu0)
        
        # Calcular incerteza
        u_e_over_m = calculate_uncertainty(
            V, u_V, I, u_I, R_cm, u_R_cm, 
            r_bobina, u_r_bobina, N, 0, k, 0, mu0, 0, 
            e_over_m
        )
        
        # Diferença percentual do valor de referência
        diff_percent = abs(e_over_m - e_over_m_ref) / e_over_m_ref * 100
        
        results.append({
            'V_fixo': V,
            'I': I,
            'u_I': u_I,
            'R_cm': R_cm,
            'u_R_cm': u_R_cm,
            'e_over_m': e_over_m,
            'u_e_over_m': u_e_over_m,
            'diff_percent': diff_percent
        })
    
    # Criar DataFrame
    df_processed = pd.DataFrame(results)
    
    # Salvar
    output_path = Path("Data/processed_V_fixo.csv")
    df_processed.to_csv(output_path, index=False)
    
    print(f"\n✅ Processamento concluído!")
    print(f"   Arquivo salvo: {output_path}")
    print(f"\n📊 Resumo:")
    print(f"   Tensões fixas: {df_processed['V_fixo'].unique()}")
    print(f"   Corrente (I): {df_processed['I'].min():.3f} - {df_processed['I'].max():.3f} A")
    print(f"   Raio (R): {df_processed['R_cm'].min():.0f} - {df_processed['R_cm'].max():.0f} cm")
    print(f"   e/m médio: {df_processed['e_over_m'].mean():.3e} C/kg")
    print(f"   Diferença média do ref: {df_processed['diff_percent'].mean():.2f}%")
    
    return df_processed

def process_I_fixo():
    """
    Processa dados de I_fixo.csv (corrente fixa, V e R variáveis).
    """
    print("\n" + "="*70)
    print("PROCESSANDO: I_fixo.csv (Corrente Fixa)")
    print("="*70)
    
    # Carregar dados
    df = pd.read_csv("Data/I_fixo.csv")
    
    print(f"\n📂 Dados carregados: {len(df)} medidas")
    print(f"\nColunas: {list(df.columns)}")
    
    # Processar cada linha
    results = []
    
    for idx, row in df.iterrows():
        # Extrair valores (convertendo vírgula para ponto)
        I = parse_csv_value(row['I fixo (A)'])
        V = parse_csv_value(row['Tensão (V)'])
        R_cm = float(row['R (cm)'])
        
        # Determinar incertezas
        u_I = get_uncertainty_from_value(I)
        u_V = get_uncertainty_from_value(V)
        u_R_cm = 0.5  # Incerteza fixa de 0.5 cm para o raio
        
        # Calcular e/m
        e_over_m = calculate_e_over_m(V, I, R_cm, r_bobina, N, k, mu0)
        
        # Calcular incerteza
        u_e_over_m = calculate_uncertainty(
            V, u_V, I, u_I, R_cm, u_R_cm, 
            r_bobina, u_r_bobina, N, 0, k, 0, mu0, 0, 
            e_over_m
        )
        
        # Diferença percentual do valor de referência
        diff_percent = abs(e_over_m - e_over_m_ref) / e_over_m_ref * 100
        
        results.append({
            'I_fixo': I,
            'V': V,
            'u_V': u_V,
            'R_cm': R_cm,
            'u_R_cm': u_R_cm,
            'e_over_m': e_over_m,
            'u_e_over_m': u_e_over_m,
            'diff_percent': diff_percent
        })
    
    # Criar DataFrame
    df_processed = pd.DataFrame(results)
    
    # Salvar
    output_path = Path("Data/processed_I_fixo.csv")
    df_processed.to_csv(output_path, index=False)
    
    print(f"\n✅ Processamento concluído!")
    print(f"   Arquivo salvo: {output_path}")
    print(f"\n📊 Resumo:")
    print(f"   Corrente fixa: {df_processed['I_fixo'].unique()[0]:.3f} A")
    print(f"   Tensão (V): {df_processed['V'].min():.1f} - {df_processed['V'].max():.1f} V")
    print(f"   Raio (R): {df_processed['R_cm'].min():.0f} - {df_processed['R_cm'].max():.0f} cm")
    print(f"   e/m médio: {df_processed['e_over_m'].mean():.3e} C/kg")
    print(f"   Diferença média do ref: {df_processed['diff_percent'].mean():.2f}%")
    
    return df_processed

def main():
    """
    Função principal - processa ambos os datasets.
    """
    print("\n🔬 DETERMINAÇÃO DE e/m - PROCESSAMENTO DE DADOS")
    print("\n📌 Constantes:")
    print(f"   N (espiras): {N}")
    print(f"   r (bobina): {r_bobina*100:.1f} ± {u_r_bobina*100:.1f} cm")
    print(f"   μ₀: {mu0:.5e} N/A²")
    print(f"   k (Helmholtz): {k}")
    print(f"   e/m (referência): {e_over_m_ref:.5e} C/kg")
    
    # Processar V_fixo
    df_V_fixo = process_V_fixo()
    
    # Processar I_fixo
    df_I_fixo = process_I_fixo()
    
    print("\n" + "="*70)
    print("✅ PROCESSAMENTO COMPLETO!")
    print("="*70)
    print("\n📁 Arquivos gerados:")
    print("   - Data/processed_V_fixo.csv")
    print("   - Data/processed_I_fixo.csv")
    print("\n🚀 Próximo passo: Execute em_analysis.py para gráficos e tabelas")
    
    return df_V_fixo, df_I_fixo

if __name__ == "__main__":
    results = main()
