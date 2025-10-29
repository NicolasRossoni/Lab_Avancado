"""
medidor_lambdas.py

Script interativo para medir coordenadas X dos picos nos espectros do branco.
- Passe o mouse sobre o gráfico para ver a coordenada X
- Clique com o botão esquerdo para copiar a coordenada X para a área de transferência
- Pressione 'q' para sair
"""

from pathlib import Path
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pyperclip  # Para copiar para área de transferência

# =========================
# CONFIGURAÇÃO DE PATHS
# =========================
PROJECT_DIR = Path.cwd() / "3-Planck"
DATA_DIR = PROJECT_DIR / "Dados" / "ComprimentoOnda"

# =========================
# FUNÇÕES DE LEITURA
# =========================
def try_read_two_column_txt(path: Path) -> pd.DataFrame:
    """Lê arquivo TXT de duas colunas (λ, intensidade)."""
    try:
        df = pd.read_csv(path, header=None, sep=";", decimal=",", engine="python")
        if df.shape[1] >= 2:
            df = df.iloc[:, :2].astype(float)
            return df
    except Exception:
        pass
    
    try:
        df = pd.read_csv(path, header=None, sep=r"\s+", decimal=",", engine="python")
        if df.shape[1] >= 2:
            df = df.iloc[:, :2].astype(float)
            return df
    except Exception:
        pass
    
    try:
        df = pd.read_csv(path, header=None, sep=",", decimal=".", engine="python")
        if df.shape[1] >= 2:
            df = df.iloc[:, :2].astype(float)
            return df
    except Exception:
        pass
    
    xs, ys = [], []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if ";" in line:
                parts = line.split(";")
            elif "," in line and " " not in line and "\t" not in line:
                parts = line.split(",")
            else:
                parts = re.split(r"\s+", line)
            
            if len(parts) < 2:
                continue
            
            a, b = parts[0].strip(), parts[1].strip()
            a = a.replace(",", ".")
            b = b.replace(",", ".")
            try:
                xs.append(float(a))
                ys.append(float(b))
            except ValueError:
                continue
    
    if not xs or not ys:
        raise ValueError(f"Arquivo inválido ou vazio: {path}")
    df = pd.DataFrame({0: xs, 1: ys}, dtype=float)
    return df

def normalize(y: np.ndarray) -> np.ndarray:
    """Normaliza para [0, 1] pelo pico (máximo)."""
    m = np.max(y) if len(y) else 1.0
    return y / m if m != 0 else y

# =========================
# PLOTAGEM INTERATIVA
# =========================
def plot_interactive():
    """Plota espectros do branco de forma interativa."""
    
    # Labels para plotar
    labels_to_plot = ["azul0", "branco1", "branco2", "branco3", "branco4", "branco5"]
    
    # Lê os dados
    spectra = {}
    for label in labels_to_plot:
        path = DATA_DIR / f"{label}.txt"
        if path.exists():
            df = try_read_two_column_txt(path)
            x = df.iloc[:, 0].to_numpy(dtype=float)
            y = df.iloc[:, 1].to_numpy(dtype=float)
            idx = np.argsort(x)
            x, y = x[idx], y[idx]
            spectra[label] = (x, normalize(y))
    
    if not spectra:
        print("Erro: Nenhum arquivo encontrado!")
        return
    
    # Cria figura
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Plota todas as curvas
    for label, (x, y) in spectra.items():
        ax.plot(x, y, label=label, alpha=0.8, linewidth=1.5)
    
    ax.set_xlabel("Comprimento de onda (nm)", fontsize=12)
    ax.set_ylabel("Intensidade normalizada (a.u.)", fontsize=12)
    ax.set_title("Medidor de Lambdas - Clique para copiar coordenada X", fontsize=14, fontweight='bold')
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(title="Laser", frameon=True, loc='upper right')
    
    # Linha vertical para indicar posição do mouse
    vline = ax.axvline(x=0, color='red', linestyle='--', linewidth=1, alpha=0, visible=False)
    
    # Texto para mostrar coordenada
    coord_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, 
                         fontsize=12, verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    def on_move(event):
        """Atualiza display quando o mouse se move."""
        if event.inaxes == ax and event.xdata is not None:
            vline.set_xdata([event.xdata, event.xdata])
            vline.set_alpha(0.5)
            vline.set_visible(True)
            coord_text.set_text(f'X = {event.xdata:.2f} nm')
            fig.canvas.draw_idle()
    
    def on_click(event):
        """Copia coordenada X quando clica."""
        if event.inaxes == ax and event.xdata is not None and event.button == 1:
            x_coord = f"{event.xdata:.2f}"
            pyperclip.copy(x_coord)
            print(f"✓ Copiado: {x_coord} nm")
            coord_text.set_text(f'✓ COPIADO: X = {x_coord} nm')
            fig.canvas.draw_idle()
    
    def on_key(event):
        """Fecha a janela ao pressionar 'q'."""
        if event.key == 'q':
            plt.close(fig)
    
    # Conecta eventos
    fig.canvas.mpl_connect('motion_notify_event', on_move)
    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('key_press_event', on_key)
    
    plt.tight_layout()
    print("\n" + "="*60)
    print("INSTRUÇÕES:")
    print("- Passe o mouse sobre o gráfico para ver a coordenada X")
    print("- Clique com o botão ESQUERDO para COPIAR a coordenada X")
    print("- Pressione 'q' para SAIR")
    print("="*60 + "\n")
    
    plt.show()

if __name__ == "__main__":
    plot_interactive()
