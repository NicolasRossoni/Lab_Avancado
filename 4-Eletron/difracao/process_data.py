import pandas as pd
import numpy as np
import os

def process_diffraction_data():
    """
    Processa os dados de difração do arquivo computational.csv
    
    Fluxo de processamento:
    1. Carrega dados do CSV
    2. Adiciona 621 aos valores R1, R2, r1, r2 (correção de coordenadas)
    3. Calcula centro médio: CM_final = (CM1 + CM2) / 2
    4. Ajusta valores relativos ao centro: subtraindo CM_final
    5. Calcula médias e incertezas para r e R
    6. Gera arquivo processed.csv
    """
    
    # Caminho para os arquivos
    data_path = "Data/computational.csv"
    output_path = "Data/processed.csv"
    
    print("Carregando dados do arquivo computational.csv...")
    
    # Carregar dados
    df = pd.read_csv(data_path)
    
    print(f"Dados carregados: {len(df)} linhas")
    print("Colunas:", df.columns.tolist())
    
    # Etapa 1: Adicionar 621 aos valores R1, R2, r1, r2 (correção de coordenadas)
    print("\nEtapa 1: Adicionando correção de coordenadas (+621)...")
    df_processed = df.copy()
    
    # Adicionar 621 a todos os valores de r e R
    df_processed['r1'] = df['r1'] + 621
    df_processed['r2'] = df['r2'] + 621
    df_processed['R1'] = df['R1'] + 621
    df_processed['R2'] = df['R2'] + 621
    
    # Etapa 2: Calcular centro médio
    print("\nEtapa 2: Calculando centro médio...")
    df_processed['CM_final'] = (df['CM1'] + df['CM2']) / 2
    
    print(f"Centro médio calculado: CM_final = {df_processed['CM_final'].iloc[0]:.1f}")
    
    # Etapa 3: Ajustar valores relativos ao centro
    print("\nEtapa 3: Ajustando valores relativos ao centro...")
    df_processed['r1_adj'] = df_processed['r1'] - df_processed['CM_final']
    df_processed['r2_adj'] = df_processed['r2'] - df_processed['CM_final']
    df_processed['R1_adj'] = df_processed['R1'] - df_processed['CM_final']
    df_processed['R2_adj'] = df_processed['R2'] - df_processed['CM_final']
    
    # Etapa 4: Calcular valores finais e incertezas
    print("\nEtapa 4: Calculando médias e incertezas...")
    
    # Criar dataframe final
    df_final = pd.DataFrame()
    
    # Copiar voltagem
    df_final['Volts'] = df['Volts']
    
    # Calcular médias dos raios (r minúsculo e R maiúsculo)
    df_final['r'] = (df_processed['r1_adj'] + df_processed['r2_adj']) / 2
    df_final['R'] = (df_processed['R1_adj'] + df_processed['R2_adj']) / 2
    
    # Calcular incertezas (metade da diferença absoluta)
    df_final['delta_r'] = np.abs(df_processed['r1_adj'] - df_processed['r2_adj']) / 2
    df_final['delta_R'] = np.abs(df_processed['R1_adj'] - df_processed['R2_adj']) / 2
    
    # Mostrar algumas estatísticas
    print(f"\nResumo dos resultados:")
    print(f"Faixa de voltagem: {df_final['Volts'].min()} - {df_final['Volts'].max()} V")
    print(f"Raio menor (r): {df_final['r'].min():.2f} a {df_final['r'].max():.2f}")
    print(f"Raio maior (R): {df_final['R'].min():.2f} a {df_final['R'].max():.2f}")
    print(f"Incerteza média em r: {df_final['delta_r'].mean():.3f}")
    print(f"Incerteza média em R: {df_final['delta_R'].mean():.3f}")
    
    # Salvar arquivo processado
    print(f"\nSalvando dados processados em {output_path}...")
    df_final.to_csv(output_path, index=False)
    
    print("Processamento concluído!")
    print(f"Arquivo {output_path} gerado com sucesso.")
    
    # Mostrar primeiras linhas do resultado
    print("\nPrimeiras 5 linhas do arquivo processado:")
    print(df_final.head())
    
    return df_final

if __name__ == "__main__":
    # Verificar se estamos no diretório correto
    if not os.path.exists("Data/computational.csv"):
        print("Erro: Arquivo Data/computational.csv não encontrado!")
        print("Certifique-se de executar o script na pasta de difração.")
    else:
        result = process_diffraction_data()
