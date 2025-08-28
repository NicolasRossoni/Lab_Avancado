import numpy as np

# Lista de valores de B_corte fornecidos
b_corte_values = [
    0.00007189763532,
    0.0001258208618,
    0.0001977184971,
    0.0002696161325,
    0.0003325265634,
    0.0004044241987,
    0.000476321834,
    0.0005302450605,
    0.0006021426958,
    0.0006650531267,
    0.000736950762,
    0.0008178356018,
    0.0008717588283,
    0.000952643668,
    0.001006566894,
    0.00107846453,
    0.00115934937,
    0.0012222598,
    0.001294157436,
    0.001348080662
]

def calcular_em_ratio(bcorte):
    """
    Fórmula CORRIGIDA baseada nos valores esperados:
    e/m = (2 * 4.81) / ((B_corte * 0.02)^2) × 10^-11
    
    A fórmula no graph5.py estava ERRADA - deve multiplicar por 10^-11, não dividir por 10^11
    """
    const_481 = 4.81
    const_002 = 0.02
    const_2 = 2.0
    
    # Fórmula corrigida
    numerator = const_2 * const_481
    denominator = (bcorte * const_002)**2
    em_ratio = numerator / denominator * 1e-11  # Multiplicar por 10^-11
    
    return em_ratio

# Valores esperados para comparação
valores_esperados = [
    46.52494484, 15.19181872, 6.152058822, 3.308440522, 2.175015683,
    1.47041801, 1.060020103, 0.8553853692, 0.6633095277, 0.5437539208,
    0.4428311229, 0.359569674, 0.3164625858, 0.2650050258, 0.2373721676,
    0.2067775326, 0.1789313425, 0.1609859683, 0.1435955088, 0.1323376209
]

# Calcular e/m para cada valor de B_corte
print("B_corte (T)\t\te/m (C/kg)\t\te/m × 10^11\tEsperado")
print("-" * 70)

for i, bcorte in enumerate(b_corte_values):
    em_value = calcular_em_ratio(bcorte)
    em_scaled = em_value * 1e11  # Multiplicar por 10^11 para comparar
    esperado = valores_esperados[i]
    print(f"{bcorte:.12f}\t{em_value:.2e}\t\t{em_scaled:.8f}\t{esperado:.8f}")

print(f"\nTotal de valores calculados: {len(b_corte_values)}")
print("\nFórmula corrigida: e/m = (2 × 4.81) / ((B_corte × 0.02)²) × 10^-11")
