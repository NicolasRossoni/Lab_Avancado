"""
analise_interativa_fix.py

Corrige os dados do computational.csv convertendo valores de centímetros para pixels.

O problema:
- No computational.csv, CM1 e CM2 estão em PIXELS
- Mas r1, r2, R1, R2 estão em CENTÍMETROS (relativos ao centro)
- O process_data.py estava somando diretamente valores em cm com pixels (ERRO!)

A conversão correta:
- pixel = centro + (cm × PIXEL_POR_CM)
- onde PIXEL_POR_CM = 80 (definido no analise_interativa.py)
"""

import pandas as pd
import numpy as np

# Constantes do analise_interativa.py
PIXEL_POR_CM = 80
CM_POR_PIXEL = 1/PIXEL_POR_CM

# Escala de conversão pixel → metro (calibrada com dados do paquímetro)
# Baseado nos valores esperados fornecidos pelo usuário
METRO_POR_PIXEL = 0.00144  # ~1.44 mm/px

def fix_computational_data():
    """
    Converte os dados do computational.csv de centímetros para pixels.
    
    Processo:
    1. Lê computational.csv
    2. Calcula centro médio: CM_final = (CM1 + CM2) / 2
    3. Converte r1, r2, R1, R2 de cm para pixels: pixel = centro + (cm × 80)
    4. Gera ProcessedData_Fixed.csv com valores corretos em pixels
    """
    
    # Caminhos
    input_path = "Data/computational.csv"
    output_path = "Data/ProcessedData_Fixed.csv"
    
    print("="*60)
    print("CORREÇÃO DOS DADOS DE DIFRAÇÃO")
    print("="*60)
    print(f"\n📂 Lendo arquivo: {input_path}")
    
    # Carregar dados
    df = pd.read_csv(input_path)
    
    print(f"✓ Dados carregados: {len(df)} linhas")
    print(f"✓ Colunas: {df.columns.tolist()}")
    
    # Mostrar exemplo dos dados originais
    print("\n📊 Exemplo dos dados ORIGINAIS (primeiras 3 linhas):")
    print(df.head(3).to_string())
    
    # Verificar valores
    print("\n🔍 Análise dos dados:")
    print(f"  CM1: {df['CM1'].iloc[0]:.1f} px (todos os valores são iguais: {df['CM1'].nunique() == 1})")
    print(f"  CM2: {df['CM2'].iloc[0]:.1f} px (todos os valores são iguais: {df['CM2'].nunique() == 1})")
    print(f"  r1 range: {df['r1'].min():.3f} a {df['r1'].max():.3f} cm")
    print(f"  R1 range: {df['R1'].min():.3f} a {df['R1'].max():.3f} cm")
    print(f"  r2 range: {df['r2'].min():.3f} a {df['r2'].max():.3f} cm")
    print(f"  R2 range: {df['R2'].min():.3f} a {df['R2'].max():.3f} cm")
    
    # Criar dataframe de saída
    df_fixed = pd.DataFrame()
    df_fixed['Volts'] = df['Volts']
    
    # Calcular centro médio
    df_fixed['CM_final'] = (df['CM1'] + df['CM2']) / 2
    centro = df_fixed['CM_final'].iloc[0]
    
    print(f"\n✓ Centro médio calculado: {centro:.1f} px")
    
    # CONVERSÃO CORRETA: pixel = centro + (cm × 80)
    print(f"\n🔧 Aplicando conversão: pixel = {centro:.1f} + (cm × {PIXEL_POR_CM})")
    
    df_fixed['r1_px'] = centro + (df['r1'] * PIXEL_POR_CM)
    df_fixed['r2_px'] = centro + (df['r2'] * PIXEL_POR_CM)
    df_fixed['R1_px'] = centro + (df['R1'] * PIXEL_POR_CM)
    df_fixed['R2_px'] = centro + (df['R2'] * PIXEL_POR_CM)
    
    # Ajustar valores relativos ao centro (em pixels)
    df_fixed['r1_adj'] = df_fixed['r1_px'] - df_fixed['CM_final']
    df_fixed['r2_adj'] = df_fixed['r2_px'] - df_fixed['CM_final']
    df_fixed['R1_adj'] = df_fixed['R1_px'] - df_fixed['CM_final']
    df_fixed['R2_adj'] = df_fixed['R2_px'] - df_fixed['CM_final']
    
    # Calcular médias e incertezas
    df_fixed['r'] = (df_fixed['r1_adj'] + df_fixed['r2_adj']) / 2
    df_fixed['R'] = (df_fixed['R1_adj'] + df_fixed['R2_adj']) / 2
    df_fixed['delta_r'] = np.abs(df_fixed['r1_adj'] - df_fixed['r2_adj']) / 2
    df_fixed['delta_R'] = np.abs(df_fixed['R1_adj'] - df_fixed['R2_adj']) / 2
    
    # Mostrar resultados
    print("\n📊 Exemplo dos dados CORRIGIDOS (primeiras 3 linhas):")
    print(df_fixed[['Volts', 'r1_px', 'R1_px', 'r2_px', 'R2_px']].head(3).to_string())
    
    print("\n📏 Verificação da conversão (primeira linha):")
    print(f"  r1: {df['r1'].iloc[0]:.3f} cm → {df_fixed['r1_px'].iloc[0]:.1f} px")
    print(f"  R1: {df['R1'].iloc[0]:.3f} cm → {df_fixed['R1_px'].iloc[0]:.1f} px")
    print(f"  r2: {df['r2'].iloc[0]:.3f} cm → {df_fixed['r2_px'].iloc[0]:.1f} px")
    print(f"  R2: {df['R2'].iloc[0]:.3f} cm → {df_fixed['R2_px'].iloc[0]:.1f} px")
    
    print("\n📈 Estatísticas dos raios ajustados (relativos ao centro):")
    print(f"  r: {df_fixed['r'].min():.2f} a {df_fixed['r'].max():.2f} px")
    print(f"  R: {df_fixed['R'].min():.2f} a {df_fixed['R'].max():.2f} px")
    print(f"  Incerteza média em r: {df_fixed['delta_r'].mean():.3f} px")
    print(f"  Incerteza média em R: {df_fixed['delta_R'].mean():.3f} px")
    
    # Criar DataFrame final para salvar
    df_output = pd.DataFrame()
    
    # Voltagem real (dividir por 10)
    df_output['Volts_real'] = df_fixed['Volts'] / 10
    df_output['Volts_CSV'] = df_fixed['Volts']
    
    # Valores em pixels
    df_output['r_px'] = df_fixed['r']
    df_output['R_px'] = df_fixed['R']
    
    # Conversão para metros
    df_output['r_m'] = df_fixed['r'] * METRO_POR_PIXEL
    df_output['R_m'] = df_fixed['R'] * METRO_POR_PIXEL
    df_output['delta_r_m'] = df_fixed['delta_r'] * METRO_POR_PIXEL
    df_output['delta_R_m'] = df_fixed['delta_R'] * METRO_POR_PIXEL
    
    # Salvar arquivo detalhado
    print(f"\n💾 Salvando dados corrigidos em: {output_path}")
    df_output.to_csv(output_path, index=False)
    
    # Criar arquivo final no formato do processed.csv antigo (compatibilidade)
    output_final_path = "Data/ProcessedData_Final.csv"
    df_final = pd.DataFrame()
    df_final['Volts'] = df_output['Volts_real']
    df_final['r'] = df_output['r_m']
    df_final['R'] = df_output['R_m']
    df_final['delta_r'] = df_output['delta_r_m']
    df_final['delta_R'] = df_output['delta_R_m']
    
    print(f"💾 Salvando arquivo final em: {output_final_path}")
    df_final.to_csv(output_final_path, index=False)
    
    print("\n✅ Processamento concluído com sucesso!")
    print(f"✅ Arquivo {output_path} gerado.")
    print(f"✅ Arquivo {output_final_path} gerado (formato compatível).")
    
    # Mostrar primeiras linhas do resultado final
    print("\n📋 Primeiras 10 linhas do arquivo detalhado:")
    print(df_output[['Volts_real', 'Volts_CSV', 'r_px', 'r_m', 'R_px', 'R_m']].head(10).to_string())
    
    print("\n📋 Primeiras 10 linhas do arquivo final (formato compatível):")
    print(df_final.head(10).to_string())
    
    print("\n📊 Valores específicos para validação:")
    print("Volts_real | r_m (metros) | Esperado")
    print("-" * 45)
    for v_csv, v_real, r_expected in [(50, 5.0, 0.116), (45, 4.5, 0.120), (40, 4.0, 0.125), 
                                        (35, 3.5, 0.138), (30, 3.0, 0.150)]:
        row = df_output[df_output['Volts_CSV'] == v_csv]
        if not row.empty:
            r_calc = row['r_m'].iloc[0]
            diff = abs(r_calc - r_expected) * 1000  # diferença em mm
            print(f"  {v_real:.1f}V     | {r_calc:.3f} m      | {r_expected:.3f} m (Δ={diff:.1f}mm)")
    
    # Comparação com o método errado (process_data.py)
    print("\n" + "="*60)
    print("COMPARAÇÃO: Método ERRADO vs CORRETO")
    print("="*60)
    
    # Método errado (do process_data.py)
    r1_errado = df['r1'].iloc[0] + 621
    R1_errado = df['R1'].iloc[0] + 621
    
    # Método correto
    r1_correto = df_fixed['r1_px'].iloc[0]
    R1_correto = df_fixed['R1_px'].iloc[0]
    
    print(f"\nPrimeira linha (15V):")
    print(f"  r1 original: {df['r1'].iloc[0]:.3f} cm")
    print(f"  └─ Método ERRADO (+ 621):  {r1_errado:.1f} px")
    print(f"  └─ Método CORRETO (× 80):  {r1_correto:.1f} px")
    print(f"  └─ Diferença: {abs(r1_correto - r1_errado):.1f} px")
    print(f"\n  R1 original: {df['R1'].iloc[0]:.3f} cm")
    print(f"  └─ Método ERRADO (+ 621):  {R1_errado:.1f} px")
    print(f"  └─ Método CORRETO (× 80):  {R1_correto:.1f} px")
    print(f"  └─ Diferença: {abs(R1_correto - R1_errado):.1f} px")
    
    print("\n" + "="*60)
    
    return df_fixed

if __name__ == "__main__":
    import os
    
    # Verificar se estamos no diretório correto
    if not os.path.exists("Data/computational.csv"):
        print("❌ Erro: Arquivo Data/computational.csv não encontrado!")
        print("   Certifique-se de executar o script na pasta 4-Eletron/difracao/")
    else:
        result = fix_computational_data()
