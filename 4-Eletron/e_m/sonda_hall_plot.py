"""
sonda_hall_plot.py

Gráfico simples do campo magnético medido pela sonda Hall vs raio.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def parse_csv_value(value):
    """Converte valores com vírgula para float."""
    if isinstance(value, str):
        return float(value.replace(',', '.'))
    return float(value)

# Carregar dados
df = pd.read_csv('Data/Sonda_hall.csv')

# Converter valores
R = df['R (cm)'].values
B = np.array([parse_csv_value(val) for val in df['B da sonda'].values])

print("="*70)
print("ANÁLISE DA SONDA HALL")
print("="*70)
print(f"\n📊 Dados carregados: {len(R)} pontos")
print(f"   R: {R.min()} - {R.max()} cm")
print(f"   B: {B.min():.2f} - {B.max():.2f} mT")

# Criar gráfico
fig, ax = plt.subplots(figsize=(12, 8))

# Scatter plot
ax.scatter(R, B, s=100, color='#2E86AB', edgecolor='black', 
           linewidth=1.5, zorder=3, label='Medidas')

# Linha conectando os pontos
ax.plot(R, B, '--', color='#2E86AB', linewidth=2, alpha=0.6, zorder=2)

# Configurações
ax.set_xlabel('Raio R (cm)', fontsize=14, fontweight='bold')
ax.set_ylabel('Campo Magnético B (mT)', fontsize=14, fontweight='bold')
ax.set_title('Teste da Sonda Hall', fontsize=16, fontweight='bold', pad=15)

ax.grid(True, alpha=0.3, linestyle=':', linewidth=1)
ax.legend(fontsize=12, framealpha=0.95, shadow=True)

# Salvar
output_path = Path('Graficos')
output_path.mkdir(exist_ok=True)
output_file = output_path / 'sonda_hall.png'

plt.tight_layout()
fig.savefig(output_file, dpi=300, bbox_inches='tight')
plt.close(fig)

print(f"\n✅ Gráfico salvo em: {output_file}")
print("="*70)
