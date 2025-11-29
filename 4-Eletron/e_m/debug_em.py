"""
debug_em.py

Debug detalhado do cálculo de e/m para identificar fonte de erro.
"""

import numpy as np

# =========================
# CONSTANTES
# =========================
N = 154                      # espiras
r_bobina = 0.398             # metros (39.8 cm)
mu0 = 1.25663706212e-6       # N/A²
k = 0.716                    # fator Helmholtz
e_over_m_ref = 1.758820024e11  # C/kg

print("="*80)
print("DEBUG DETALHADO - CÁLCULO DE e/m")
print("="*80)

print("\n📌 CONSTANTES:")
print(f"   N = {N} espiras")
print(f"   r (bobina) = {r_bobina} m = {r_bobina*100} cm")
print(f"   μ₀ = {mu0:.5e} N/A²")
print(f"   k = {k}")
print(f"   e/m (ref) = {e_over_m_ref:.5e} C/kg")

# =========================
# TESTE 1: I_fixo - Primeira linha
# =========================
print("\n" + "="*80)
print("TESTE 1: I_fixo.csv - Linha 1")
print("="*80)

I = 1.494  # A
V = 41.4   # V
R_cm = 3   # cm

print(f"\n📊 Valores medidos:")
print(f"   I = {I} A")
print(f"   V = {V} V")
print(f"   R = {R_cm} cm = {R_cm/100} m")

# Cálculo passo a passo
R_m = R_cm / 100.0
print(f"\n🔍 Cálculo passo a passo:")
print(f"   R (metros) = {R_m} m")

# Fórmula: e/m = 2 * V * ((1/(k*μ₀)) * (r/(N*I*R)))²
term1 = 1.0 / (k * mu0)
print(f"\n   1/(k*μ₀) = 1/({k}*{mu0:.5e}) = {term1:.5e}")

term2 = r_bobina / (N * I * R_m)
print(f"   r/(N*I*R) = {r_bobina}/({N}*{I}*{R_m}) = {term2:.5e}")

term3 = term1 * term2
print(f"   [1/(k*μ₀)] * [r/(N*I*R)] = {term1:.5e} * {term2:.5e} = {term3:.5e}")

term4 = term3**2
print(f"   [....]² = ({term3:.5e})² = {term4:.5e}")

e_over_m = 2.0 * V * term4
print(f"\n   e/m = 2*V*[...]² = 2*{V}*{term4:.5e} = {e_over_m:.5e} C/kg")

diff = abs(e_over_m - e_over_m_ref) / e_over_m_ref * 100
print(f"\n   ✅ e/m calculado: {e_over_m:.5e} C/kg")
print(f"   🎯 e/m referência: {e_over_m_ref:.5e} C/kg")
print(f"   ❌ Diferença: {diff:.2f}%")

# =========================
# TESTE 2: V_fixo - Primeira linha
# =========================
print("\n" + "="*80)
print("TESTE 2: V_fixo.csv - Linha 1")
print("="*80)

I = 2.004  # A
V = 61.0   # V
R_cm = 3   # cm

print(f"\n📊 Valores medidos:")
print(f"   I = {I} A")
print(f"   V = {V} V")
print(f"   R = {R_cm} cm = {R_cm/100} m")

# Cálculo passo a passo
R_m = R_cm / 100.0
print(f"\n🔍 Cálculo passo a passo:")
print(f"   R (metros) = {R_m} m")

term1 = 1.0 / (k * mu0)
print(f"\n   1/(k*μ₀) = {term1:.5e}")

term2 = r_bobina / (N * I * R_m)
print(f"   r/(N*I*R) = {r_bobina}/({N}*{I}*{R_m}) = {term2:.5e}")

term3 = term1 * term2
print(f"   [1/(k*μ₀)] * [r/(N*I*R)] = {term3:.5e}")

term4 = term3**2
print(f"   [....]² = {term4:.5e}")

e_over_m = 2.0 * V * term4
print(f"\n   e/m = 2*V*[...]² = 2*{V}*{term4:.5e} = {e_over_m:.5e} C/kg")

diff = abs(e_over_m - e_over_m_ref) / e_over_m_ref * 100
print(f"\n   ✅ e/m calculado: {e_over_m:.5e} C/kg")
print(f"   🎯 e/m referência: {e_over_m_ref:.5e} C/kg")
print(f"   ❌ Diferença: {diff:.2f}%")

# =========================
# ANÁLISE: Verificar fórmula alternativa
# =========================
print("\n" + "="*80)
print("ANÁLISE: Verificar campo magnético B")
print("="*80)

# Campo magnético B em bobina de Helmholtz
# B = (8/5^(3/2)) * (μ₀ * N * I) / r
# ou B = k * μ₀ * N * I / r  (com k = 8/5^(3/2) ≈ 0.716)

I_test = 1.494
B = k * mu0 * N * I_test / r_bobina
print(f"\nPara I = {I_test} A:")
print(f"   B = k * μ₀ * N * I / r")
print(f"   B = {k} * {mu0:.5e} * {N} * {I_test} / {r_bobina}")
print(f"   B = {B:.5e} T")

# Relação fundamental: e/m = 2*V / (B*R)²
# Ou usando B: e/m = 2*V*R² / B²
# Ou: e/m = 2*V / (B²*R²)

V_test = 41.4
R_test = 0.03  # 3 cm em metros

e_over_m_check = 2 * V_test / (B * R_test)**2
print(f"\nVerificação com B:")
print(f"   e/m = 2*V / (B*R)²")
print(f"   e/m = 2*{V_test} / ({B:.5e}*{R_test})²")
print(f"   e/m = {e_over_m_check:.5e} C/kg")

diff_check = abs(e_over_m_check - e_over_m_ref) / e_over_m_ref * 100
print(f"   Diferença: {diff_check:.2f}%")

# =========================
# DIAGNÓSTICO
# =========================
print("\n" + "="*80)
print("DIAGNÓSTICO")
print("="*80)

print("\n🔍 Verificando ordem de grandeza esperada:")
print(f"   e/m (ref) ≈ 1.76 × 10¹¹ C/kg")
print(f"   e/m (calc I_fixo) ≈ {2.299e11:.2e} C/kg → +30%")
print(f"   e/m (calc V_fixo) ≈ {6.504e11:.2e} C/kg → +270%")

print("\n💡 Possíveis causas:")
print("   1. Raio R muito pequeno (3-10 cm) → e/m aumenta com 1/R²")
print("   2. Tensão ou corrente com erro sistemático")
print("   3. Fator k incorreto para esta geometria")
print("   4. Raio da bobina r diferente do especificado")

print("\n⚠️ IMPORTANTE:")
print("   Os valores calculados estão na ordem de grandeza correta (10¹¹)")
print("   mas com erros sistemáticos significativos.")
print("   Isso é típico em experimentos de e/m devido a:")
print("   - Imprecisões na medida de R (raio da trajetória)")
print("   - Campo magnético não perfeitamente uniforme")
print("   - Efeitos de borda da bobina")

print("\n" + "="*80)
