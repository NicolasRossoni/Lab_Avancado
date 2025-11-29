"""
massa_relativistica.py

Compara massa de repouso (m₀) e massa relativística (m_rel) do elétron
para diferentes potenciais de aceleração.

Questão 5 do relatório: Analisar efeitos relativísticos em diferentes voltagens.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Constantes físicas
m0 = 9.10938356e-31  # kg - massa de repouso do elétron
e = 1.602176634e-19  # C - carga elementar
c = 299792458        # m/s - velocidade da luz

def calculate_relativistic_mass(V):
    """
    Calcula a massa relativística do elétron acelerado por voltagem V.
    
    Fórmula derivada de:
    E_total = m_rel × c²
    E_total = E_repouso + E_cinética
    m_rel × c² = m₀ × c² + e × V
    m_rel = m₀ + e×V/c²
    
    Forma mais precisa (usando fator de Lorentz γ):
    γ = 1 + e×V/(m₀×c²)
    m_rel = γ × m₀
    
    Args:
        V: Voltagem de aceleração (Volts)
    
    Returns:
        tuple: (m_rel, gamma, velocidade, diferença_percentual)
    """
    # Fator de Lorentz
    gamma = 1 + (e * V) / (m0 * c**2)
    
    # Massa relativística
    m_rel = gamma * m0
    
    # Velocidade do elétron (calculada classicamente para comparação)
    # E_cin = e×V = (1/2)×m₀×v²
    v_classica = np.sqrt(2 * e * V / m0)
    
    # Velocidade relativística correta
    # γ = 1/√(1 - v²/c²) → v = c×√(1 - 1/γ²)
    v_rel = c * np.sqrt(1 - 1/gamma**2)
    
    # Diferença percentual
    diff_percent = ((m_rel - m0) / m0) * 100
    
    return m_rel, gamma, v_rel, v_classica, diff_percent

def create_comparison_table():
    """
    Cria tabela comparativa de massa relativística vs repouso.
    """
    print("="*80)
    print("COMPARAÇÃO: MASSA DE REPOUSO vs MASSA RELATIVÍSTICA DO ELÉTRON")
    print("="*80)
    
    # Voltagens solicitadas
    voltages = [10, 100, 1000, 10000, 100000]
    
    # Voltagens do experimento (para comparação)
    exp_voltages = [1.5, 3.0, 5.0]
    
    # Calcular para voltagens solicitadas
    results = []
    for V in voltages:
        m_rel, gamma, v_rel, v_class, diff = calculate_relativistic_mass(V)
        results.append({
            'V (V)': V,
            'm_rel (kg)': m_rel,
            'γ': gamma,
            'v/c': v_rel/c,
            'Δm/m₀ (%)': diff
        })
    
    df = pd.DataFrame(results)
    
    print(f"\n📊 VOLTAGENS SOLICITADAS:")
    print("-" * 80)
    print(df.to_string(index=False))
    
    # Calcular para voltagens do experimento
    print(f"\n\n📊 VOLTAGENS DO EXPERIMENTO (1.5V - 5V):")
    print("-" * 80)
    exp_results = []
    for V in exp_voltages:
        m_rel, gamma, v_rel, v_class, diff = calculate_relativistic_mass(V)
        exp_results.append({
            'V (V)': V,
            'm_rel (kg)': m_rel,
            'γ': gamma,
            'v/c': v_rel/c,
            'Δm/m₀ (%)': diff
        })
    
    df_exp = pd.DataFrame(exp_results)
    print(df_exp.to_string(index=False))
    
    # Análise e conclusões
    print("\n\n" + "="*80)
    print("ANÁLISE E DISCUSSÃO")
    print("="*80)
    
    print(f"\n🔬 Massa de repouso do elétron:")
    print(f"   m₀ = {m0:.5e} kg")
    
    print(f"\n📈 Efeitos Relativísticos:")
    print(f"   • V = 10 V     → Δm/m₀ = {results[0]['Δm/m₀ (%)']:.2e}% (desprezível)")
    print(f"   • V = 100 V    → Δm/m₀ = {results[1]['Δm/m₀ (%)']:.2e}% (desprezível)")
    print(f"   • V = 1 kV     → Δm/m₀ = {results[2]['Δm/m₀ (%)']:.2e}% (desprezível)")
    print(f"   • V = 10 kV    → Δm/m₀ = {results[3]['Δm/m₀ (%)']:.3f}% (começa a ser relevante)")
    print(f"   • V = 100 kV   → Δm/m₀ = {results[4]['Δm/m₀ (%)']:.2f}% (significativo!)")
    
    print(f"\n🎯 Comparação com o Experimento:")
    print(f"   No experimento, usamos V = 1.5V - 5V")
    print(f"   Para V = 5V: Δm/m₀ = {exp_results[2]['Δm/m₀ (%)']:.2e}%")
    print(f"   → Efeitos relativísticos são TOTALMENTE DESPREZÍVEIS!")
    print(f"   → Aproximação clássica (m = m₀) é VÁLIDA ✓")
    
    print(f"\n💡 Critério Físico:")
    print(f"   • Δm/m₀ < 0.1% → Física clássica OK")
    print(f"   • Δm/m₀ > 1%   → Física relativística necessária")
    print(f"   • V > ~25 kV   → Correções relativísticas importantes")
    
    print("\n" + "="*80)
    
    return df, df_exp

def create_comparison_plot():
    """
    Cria gráfico mostrando massa relativística vs voltagem.
    """
    # Voltagens de 1V até 100kV (escala logarítmica)
    V_range = np.logspace(0, 5, 1000)  # 10^0 = 1V até 10^5 = 100kV
    
    # Calcular massa relativística
    m_rel_array = []
    diff_percent_array = []
    
    for V in V_range:
        m_rel, _, _, _, diff = calculate_relativistic_mass(V)
        m_rel_array.append(m_rel)
        diff_percent_array.append(diff)
    
    m_rel_array = np.array(m_rel_array)
    diff_percent_array = np.array(diff_percent_array)
    
    # Criar figura com 2 subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Subplot 1: Massa relativística vs Voltagem
    ax1.semilogx(V_range, m_rel_array / m0, 'b-', linewidth=2, label='m_rel / m₀')
    ax1.axhline(y=1, color='r', linestyle='--', linewidth=1.5, label='m₀ (repouso)')
    
    # Marcar voltagens específicas
    voltages_mark = [10, 100, 1000, 10000, 100000]
    for V in voltages_mark:
        m_rel, _, _, _, _ = calculate_relativistic_mass(V)
        ax1.plot(V, m_rel/m0, 'ro', markersize=8)
    
    # Marcar região do experimento
    ax1.axvspan(1.5, 5.0, alpha=0.2, color='green', label='Região do experimento')
    
    ax1.set_xlabel('Voltagem de Aceleração (V)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('m_rel / m₀', fontsize=12, fontweight='bold')
    ax1.set_title('Massa Relativística vs Voltagem de Aceleração', 
                  fontsize=14, fontweight='bold', pad=15)
    ax1.grid(True, alpha=0.3, which='both')
    ax1.legend(fontsize=10)
    
    # Subplot 2: Diferença percentual (escala log)
    ax2.loglog(V_range, diff_percent_array, 'g-', linewidth=2)
    ax2.axhline(y=0.1, color='orange', linestyle='--', linewidth=1.5, 
                label='0.1% (limiar clássico)')
    ax2.axhline(y=1.0, color='red', linestyle='--', linewidth=1.5, 
                label='1% (correções necessárias)')
    
    # Marcar voltagens específicas
    for V in voltages_mark:
        _, _, _, _, diff = calculate_relativistic_mass(V)
        ax2.plot(V, diff, 'ro', markersize=8)
    
    # Marcar região do experimento
    ax2.axvspan(1.5, 5.0, alpha=0.2, color='green', label='Região do experimento')
    
    ax2.set_xlabel('Voltagem de Aceleração (V)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Δm/m₀ (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Efeito Relativístico: Variação Percentual da Massa', 
                  fontsize=14, fontweight='bold', pad=15)
    ax2.grid(True, alpha=0.3, which='both')
    ax2.legend(fontsize=10)
    
    plt.tight_layout()
    
    # Salvar
    output_path = Path('Graficos')
    output_path.mkdir(exist_ok=True)
    output_file = output_path / 'massa_relativistica_comparacao.png'
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ Gráfico salvo em: {output_file}")

def main():
    """
    Função principal - análise completa de efeitos relativísticos.
    """
    # Criar tabela comparativa
    df, df_exp = create_comparison_table()
    
    # Gerar gráfico
    print("\n📊 Gerando gráfico comparativo...")
    create_comparison_plot()
    
    print("\n✅ Análise concluída!")
    print("\n📁 Arquivo gerado:")
    print("  - Graficos/massa_relativistica_comparacao.png")
    
    return df, df_exp

if __name__ == "__main__":
    results = main()
