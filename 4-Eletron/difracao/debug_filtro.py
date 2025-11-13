"""
Script de debug para verificar o filtro V > 3.5
"""

import pandas as pd
import numpy as np
from scipy import stats

# Carregar dados da foto
df_foto = pd.read_csv("Data/lambda_analysis_computacional_final.csv")

print("="*60)
print("DEBUG - ANÁLISE DO FILTRO V > 3.5")
print("="*60)

# Mostrar todas as voltagens disponíveis
print(f"\nTotal de medidas: {len(df_foto)}")
print(f"Voltagens disponíveis: {sorted(df_foto['Volts'].unique())}")

# Aplicar filtro V > 3.5
V_volts = df_foto['Volts'].values
mask = V_volts > 3.5

print(f"\n📊 Filtro V > 3.5:")
print(f"  Pontos filtrados: {mask.sum()}")
print(f"  Voltagens selecionadas: {sorted(V_volts[mask])}")

# Calcular ajustes
V_inv_sqrt_filtered = V_volts[mask] ** (-0.5)
lambda_d1_filtered = df_foto['lambda_d1'].values[mask]
lambda_d2_filtered = df_foto['lambda_d2'].values[mask]

print(f"\n📈 Dados filtrados:")
print(f"  V^(-1/2): {V_inv_sqrt_filtered}")
print(f"  λ_d1: {lambda_d1_filtered}")
print(f"  λ_d2: {lambda_d2_filtered}")

# Ajuste linear
result_d1 = stats.linregress(V_inv_sqrt_filtered, lambda_d1_filtered)
result_d2 = stats.linregress(V_inv_sqrt_filtered, lambda_d2_filtered)

print(f"\n🔍 Ajuste Linear d1:")
print(f"  Coef. angular: {result_d1.slope:.3f} ± {result_d1.stderr:.3f}")
print(f"  R²: {result_d1.rvalue**2:.6f}")

print(f"\n🔍 Ajuste Linear d2:")
print(f"  Coef. angular: {result_d2.slope:.3f} ± {result_d2.stderr:.3f}")
print(f"  R²: {result_d2.rvalue**2:.6f}")

# Calcular h
m_electron = 9.11e-31
e_charge = 1.6e-19
sqrt_2me = np.sqrt(2 * m_electron * e_charge)

h_d1 = result_d1.slope * sqrt_2me * 1e-10
h_d1_unc = result_d1.stderr * sqrt_2me * 1e-10

h_d2 = result_d2.slope * sqrt_2me * 1e-10
h_d2_unc = result_d2.stderr * sqrt_2me * 1e-10

h_ref = 6.62607015e-34

print(f"\n🎯 Constante de Planck:")
print(f"  h_d1 = {h_d1:.3e} ± {h_d1_unc:.2e} J·s")
print(f"  Diferença: {abs(h_d1 - h_ref)/h_ref * 100:.1f}%")
print(f"  h_d2 = {h_d2:.3e} ± {h_d2_unc:.2e} J·s")
print(f"  Diferença: {abs(h_d2 - h_ref)/h_ref * 100:.1f}%")
print(f"  h_ref = {h_ref:.3e} J·s")

print("\n" + "="*60)
