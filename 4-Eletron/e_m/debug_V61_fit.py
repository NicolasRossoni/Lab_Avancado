"""
debug_V61_fit.py

Debug do fit para identificar o problema com V=61V
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, minimize

# Constantes
e_over_m_ref = 1.758820024e11
N = 154
r_bobina = 0.398
mu0 = 1.25663706212e-6
k = 0.716

def calculate_R_theoretical(V, I):
    """Calcula R teórico dado V e I."""
    B = k * mu0 * N * I / r_bobina
    R_m = np.sqrt(2 * V) / (B * np.sqrt(e_over_m_ref))
    R_cm = R_m * 100
    return R_cm

def R_fit_function(I, V_fit):
    """Função para fit."""
    return calculate_R_theoretical(V_fit, I)

def residuals_squared(V, I_data, R_data):
    """Calcula soma dos resíduos quadrados."""
    R_pred = np.array([calculate_R_theoretical(V, I) for I in I_data])
    return np.sum((R_data - R_pred)**2)

# Carregar dados
df = pd.read_csv("Data/processed_V_fixo.csv")
df_V61 = df[df['V_fixo'] == 61.0]

I_data = df_V61['I'].values
R_data = df_V61['R_cm'].values
u_R = df_V61['u_R_cm'].values

print("="*70)
print("DEBUG - FIT DE V=61V")
print("="*70)

print("\n📊 Dados:")
for i in range(len(I_data)):
    print(f"   I = {I_data[i]:.3f} A, R = {R_data[i]:.0f} cm")

# Teste 1: Testar range de valores de V manualmente
print("\n🔍 Teste 1: Avaliar função de custo para diferentes V")
print("-"*70)

V_test = np.linspace(1, 150, 200)
chi_squared = []

for V in V_test:
    chi2 = residuals_squared(V, I_data, R_data)
    chi_squared.append(chi2)

chi_squared = np.array(chi_squared)
V_best_manual = V_test[np.argmin(chi_squared)]

print(f"\nV com menor χ² (busca manual): {V_best_manual:.2f} V")
print(f"χ² mínimo: {np.min(chi_squared):.2f}")

# Plotar função de custo
fig1, ax1 = plt.subplots(figsize=(10, 6))
ax1.plot(V_test, chi_squared, 'b-', linewidth=2)
ax1.axvline(V_best_manual, color='r', linestyle='--', linewidth=2, 
            label=f'Mínimo: V={V_best_manual:.2f}V')
ax1.axvline(61.0, color='orange', linestyle='--', linewidth=2,
            label='Nominal: V=61.0V')
ax1.set_xlabel('V (Volts)', fontsize=12, fontweight='bold')
ax1.set_ylabel('χ² (Soma dos Resíduos²)', fontsize=12, fontweight='bold')
ax1.set_title('Função de Custo: χ² vs V', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=11)
plt.tight_layout()
plt.savefig('Graficos/debug_chi_squared.png', dpi=200)
plt.close()

print(f"✅ Gráfico salvo: Graficos/debug_chi_squared.png")

# Teste 2: Usar curve_fit
print("\n🔍 Teste 2: curve_fit (scipy)")
print("-"*70)

try:
    popt1, pcov1 = curve_fit(R_fit_function, I_data, R_data, p0=[61.0], 
                              bounds=(0, 200))
    V_fit1 = popt1[0]
    V_fit1_err = np.sqrt(pcov1[0, 0])
    print(f"V (curve_fit, bounds=[0,200]): {V_fit1:.2f} ± {V_fit1_err:.2f} V")
    print(f"χ²: {residuals_squared(V_fit1, I_data, R_data):.2f}")
except Exception as e:
    print(f"❌ Erro: {e}")

# Teste 3: Usar minimize (otimização direta)
print("\n🔍 Teste 3: minimize (scipy)")
print("-"*70)

result = minimize(lambda V: residuals_squared(V[0], I_data, R_data), 
                  x0=[61.0], bounds=[(0.1, 200)])
V_fit2 = result.x[0]
print(f"V (minimize): {V_fit2:.2f} V")
print(f"χ²: {residuals_squared(V_fit2, I_data, R_data):.2f}")
print(f"Sucesso: {result.success}")

# Teste 4: Visualizar curvas para diferentes V
print("\n📊 Teste 4: Comparação visual de curvas")
print("-"*70)

fig2, ax2 = plt.subplots(figsize=(12, 8))

# Dados experimentais
ax2.errorbar(I_data, R_data, yerr=u_R, fmt='o', color='black', 
             markersize=10, capsize=5, capthick=2, label='Dados', 
             linewidth=2, elinewidth=2, zorder=5)

# Range de I para plotar curvas
I_range = np.linspace(I_data.min()*0.7, I_data.max()*1.3, 100)

# Testar diferentes valores de V
V_values = [10, 30, 50, 61, 80, 100, V_best_manual]
colors = plt.cm.rainbow(np.linspace(0, 1, len(V_values)))

for V_val, color in zip(V_values, colors):
    R_curve = [calculate_R_theoretical(V_val, I) for I in I_range]
    chi2 = residuals_squared(V_val, I_data, R_data)
    label = f'V={V_val:.1f}V (χ²={chi2:.1f})'
    
    if V_val == V_best_manual:
        ax2.plot(I_range, R_curve, '--', color='red', linewidth=3,
                label=label + ' ★', zorder=4)
    elif V_val == 61:
        ax2.plot(I_range, R_curve, '-.', color='orange', linewidth=2.5,
                label=label + ' (nominal)', alpha=0.7, zorder=3)
    else:
        ax2.plot(I_range, R_curve, '-', color=color, linewidth=1.5,
                label=label, alpha=0.6, zorder=2)

ax2.set_xlabel('Corrente I (A)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Raio R (cm)', fontsize=12, fontweight='bold')
ax2.set_title('Comparação de Curvas Teóricas\nQual V melhor descreve os dados?',
             fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=9, loc='best', ncol=2)
plt.tight_layout()
plt.savefig('Graficos/debug_curves_comparison.png', dpi=200)
plt.close()

print(f"✅ Gráfico salvo: Graficos/debug_curves_comparison.png")

# Resumo
print("\n" + "="*70)
print("RESUMO")
print("="*70)
print(f"\n🎯 Melhor V encontrado:")
print(f"   Busca manual:     V = {V_best_manual:.2f} V")
print(f"   curve_fit:        V = {V_fit1:.2f} V")
print(f"   minimize:         V = {V_fit2:.2f} V")
print(f"   Nominal:          V = 61.00 V")

print(f"\n💡 Análise:")
print(f"   Os três métodos devem convergir para o mesmo valor.")
print(f"   Se não convergirem, há problema na função ou dados.")

print("\n" + "="*70)
