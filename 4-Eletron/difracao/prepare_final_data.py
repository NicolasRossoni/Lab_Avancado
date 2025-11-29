"""
prepare_final_data.py

Prepara os arquivos finais de dados para análise:
1. finaldata_paquimetro.csv - dados medidos com paquímetro (já corretos)
2. finaldata_computacional.csv - dados computacionais corrigidos (da análise interativa)
"""

import pandas as pd
import numpy as np

def prepare_paquimetro_data():
    """
    Prepara finaldata_paquimetro.csv combinando dados de d1 e d2.
    Formato final: Volts, r, R, delta_r, delta_R (todos em metros)
    """
    print("="*60)
    print("PREPARANDO DADOS DO PAQUÍMETRO")
    print("="*60)
    
    # Carregar dados do paquímetro
    df_d1 = pd.read_csv("Data/paquimetro_d1.csv")
    df_d2 = pd.read_csv("Data/paquimetro_d2.csv")
    
    print(f"\n📂 Dados carregados:")
    print(f"  - paquimetro_d1.csv: {len(df_d1)} linhas")
    print(f"  - paquimetro_d2.csv: {len(df_d2)} linhas")
    
    # Preparar DataFrame final
    df_final = pd.DataFrame()
    
    # Tensão (converter string "3,0" para float 3.0)
    df_final['Volts'] = df_d1['Tensão (V)'].str.replace(',', '.').astype(float)
    
    # Raio médio d1 (r minúsculo) - converter para metros
    # d_med está em CM e é um DIÂMETRO, então dividir por 2 para obter raio
    d_med_cm = df_d1['d_med'].str.replace(',', '.').astype(float)
    r_cm = d_med_cm / 2.0  # diâmetro -> raio
    df_final['r'] = r_cm / 100.0  # cm -> m
    
    # Raio médio d2 (R maiúsculo) - converter para metros
    D_med_cm = df_d2['d_med'].str.replace(',', '.').astype(float)
    R_cm = D_med_cm / 2.0  # diâmetro -> raio
    df_final['R'] = R_cm / 100.0  # cm -> m
    
    # Calcular incertezas (metade da diferença entre diâmetros, depois dividir por 2 para raio)
    d1_diam1 = df_d1['diametro1'].str.replace(',', '.').astype(float)
    d1_diam2 = df_d1['diametro2'].str.replace(',', '.').astype(float)
    delta_diam_cm = np.abs(d1_diam1 - d1_diam2) / 2.0  # incerteza do diâmetro (em cm)
    delta_r_cm = delta_diam_cm / 2.0  # incerteza do raio (em cm)
    df_final['delta_r'] = delta_r_cm / 100.0  # cm -> m
    
    d2_diam1 = df_d2['diametro1'].str.replace(',', '.').astype(float)
    d2_diam2 = df_d2['diametro2'].str.replace(',', '.').astype(float)
    delta_Diam_cm = np.abs(d2_diam1 - d2_diam2) / 2.0  # incerteza do diâmetro (em cm)
    delta_R_cm = delta_Diam_cm / 2.0  # incerteza do raio (em cm)
    df_final['delta_R'] = delta_R_cm / 100.0  # cm -> m
    
    # Salvar
    output_path = "Data/finaldata_paquimetro.csv"
    df_final.to_csv(output_path, index=False)
    
    print(f"\n✅ Arquivo gerado: {output_path}")
    print(f"\n📋 Primeiras linhas:")
    print(df_final.to_string())
    
    print(f"\n📊 Resumo:")
    print(f"  Voltagens: {df_final['Volts'].min():.1f} a {df_final['Volts'].max():.1f} V")
    print(f"  r (raio menor): {df_final['r'].min():.4f} a {df_final['r'].max():.4f} m")
    print(f"  R (raio maior): {df_final['R'].min():.4f} a {df_final['R'].max():.4f} m")
    
    return df_final

def prepare_computational_data():
    """
    Copia ProcessedData_Final.csv para finaldata_computacional.csv
    """
    print("\n" + "="*60)
    print("PREPARANDO DADOS COMPUTACIONAIS")
    print("="*60)
    
    # Carregar dados processados
    df = pd.read_csv("Data/ProcessedData_Final.csv")
    
    print(f"\n📂 Dados carregados de ProcessedData_Final.csv: {len(df)} linhas")
    
    # Salvar com novo nome
    output_path = "Data/finaldata_computacional.csv"
    df.to_csv(output_path, index=False)
    
    print(f"\n✅ Arquivo gerado: {output_path}")
    print(f"\n📋 Primeiras 5 linhas:")
    print(df.head().to_string())
    
    print(f"\n📊 Resumo:")
    print(f"  Voltagens: {df['Volts'].min():.1f} a {df['Volts'].max():.1f} V")
    print(f"  r (raio menor): {df['r'].min():.4f} a {df['r'].max():.4f} m")
    print(f"  R (raio maior): {df['R'].min():.4f} a {df['R'].max():.4f} m")
    
    return df

if __name__ == "__main__":
    import os
    
    # Verificar se estamos no diretório correto
    if not os.path.exists("Data"):
        print("❌ Erro: Pasta Data/ não encontrada!")
        print("   Execute o script na pasta 4-Eletron/difracao/")
    else:
        # Preparar ambos os datasets
        df_paq = prepare_paquimetro_data()
        df_comp = prepare_computational_data()
        
        print("\n" + "="*60)
        print("PREPARAÇÃO CONCLUÍDA!")
        print("="*60)
        print("\n📁 Arquivos gerados:")
        print("  1. Data/finaldata_paquimetro.csv")
        print("  2. Data/finaldata_computacional.csv")
        print("\n🚀 Próximo passo: Execute lambda_analysis2.py")
