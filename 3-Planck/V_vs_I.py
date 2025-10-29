"""
V_vs_I.py

Lê dados de tensão-corrente (V, I) de CSVs em 'dados/Tensao_Corrente/',
importa comprimentos de onda do Lambda_Outputs.csv e gera gráficos I vs V
agrupados por cor (5 gráficos) + 1 comparativo.
"""

from __future__ import annotations
from pathlib import Path
import re
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# CONFIGURAÇÃO DE PATHS
# =========================
PROJECT_DIR = Path.cwd() / "3-Planck"
DATA_DIR = PROJECT_DIR / "Dados" / "Tensao_Corrente"  # pasta com os .csv
LAMBDA_CSV = PROJECT_DIR / "Dados" / "Lambda_Outputs.csv"  # lambdas calculados
PLOTS_DIR = PROJECT_DIR / "Graficos" / "I_vs_V"  # onde salvar os .png

# =========================
# PARÂMETROS
# =========================
# Rótulos (labels) a compor o gráfico comparativo (edite à vontade):
COMPARISON_LABELS = ["azul1", "laranja1", "verde1", "vermelho1"]

# Constantes físicas (SI)
E_CHARGE = 1.602176634e-19  # Carga elementar em Coulombs
C_LIGHT = 299792458.0  # Velocidade da luz em m/s
H_PLANCK_REF = 6.62607015e-34  # Constante de Planck de referência em J·s

# Paleta para o gráfico comparativo (cores específicas por cor do laser):
COLOR_MAP = {
    "azul": "tab:blue",
    "branco": "black",
    "laranja": "tab:orange",
    "verde": "tab:green",
    "vermelho": "tab:red",
}

# =========================
# FUNÇÕES AUXILIARES
# =========================
def ensure_dirs() -> None:
    """Garante que as pastas necessárias existam."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

def parse_label_from_filename(path: Path) -> str:
    """
    Extrai o label a partir do nome do arquivo.
    Ex.: 'Lab Avançado 2025.2 - azul1.csv' -> 'azul1'
    Pega a última palavra antes de .csv (split por espaço)
    """
    name = path.stem  # Remove .csv
    parts = name.split()
    if parts:
        return parts[-1]  # Última palavra
    return name

def split_color_and_index(label: str) -> Tuple[str, str]:
    """
    Separa 'cor' e 'índice' do label.
    Ex.: 'azul0' -> ('azul', '0'); 'branco12' -> ('branco','12')
    Se não houver dígitos, retorna ('label', '').
    """
    m = re.match(r"^([A-Za-zÀ-ÿ]+)(\d*)$", label)
    if not m:
        return label, ""
    return m.group(1).lower(), m.group(2)

def read_lambda_mapping(lambda_csv: Path) -> Dict[str, List[float]]:
    """
    Lê o arquivo Lambda_Outputs.csv e retorna um dicionário:
      { label: [lambda1, lambda2, ...] }
    Se houver múltiplos lambdas, eles estão separados por '/'.
    """
    if not lambda_csv.exists():
        print(f"Aviso: {lambda_csv} não encontrado. Legendas sem comprimento de onda.")
        return {}
    
    df = pd.read_csv(lambda_csv)
    mapping = {}
    
    for _, row in df.iterrows():
        label = row['label']
        lambda_str = str(row['lambda'])
        
        # Separa por '/' se houver múltiplos lambdas
        if '/' in lambda_str:
            lambdas = [float(lam.strip()) for lam in lambda_str.split('/')]
        else:
            lambdas = [float(lambda_str)]
        
        mapping[label] = lambdas
    
    return mapping

def read_all_vi_data(data_dir: Path) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Lê todos os arquivos .csv e devolve um dicionário:
      { label: (V_mV, I_mA) }
    Converte corrente de 10⁻⁵A para mA (divide por 100).
    """
    vi_data: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    
    for path in sorted(data_dir.glob("*.csv")):
        label = parse_label_from_filename(path)
        
        try:
            # Lê CSV (primeira linha é cabeçalho)
            df = pd.read_csv(path)
            
            # Primeira coluna: tensão (mV)
            # Segunda coluna: corrente (10⁻⁵ A)
            V = df.iloc[:, 0].to_numpy(dtype=float)  # mV
            I_raw = df.iloc[:, 1].to_numpy(dtype=float)  # 10⁻⁵ A
            
            # Converte corrente de 10⁻⁵A para mA
            # 10⁻⁵ A = 10⁻⁵ A * (1000 mA / 1 A) = 10⁻² mA
            # Ou seja: dividir por 100
            I_mA = I_raw / 100.0
            
            vi_data[label] = (V, I_mA)
        
        except Exception as e:
            print(f"Erro ao ler {path.name}: {e}")
            continue
    
    return vi_data

def linear_regression(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """
    Regressão linear por mínimos quadrados.
    Retorna (a, b) onde y = a*x + b
    """
    n = len(x)
    if n < 2:
        return 0.0, 0.0
    
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    
    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sum((x - x_mean) ** 2)
    
    if denominator == 0:
        return 0.0, y_mean
    
    a = numerator / denominator  # coeficiente angular
    b = y_mean - a * x_mean      # coeficiente linear
    
    return a, b

def calculate_v_min_piecewise(V: np.ndarray, I: np.ndarray) -> float:
    """
    Calcula V_mínimo usando método Piecewise Linear.
    - Primeira reta: fit dos primeiros 4 pontos
    - Segunda reta: fit dos últimos 4 pontos
    - V_min: interseção das duas retas
    """
    if len(V) < 8:
        return float('nan')  # Dados insuficientes
    
    # Primeira reta: primeiros 4 pontos
    V1 = V[:4]
    I1 = I[:4]
    a1, b1 = linear_regression(V1, I1)  # I = a1*V + b1
    
    # Segunda reta: últimos 4 pontos
    V2 = V[-4:]
    I2 = I[-4:]
    a2, b2 = linear_regression(V2, I2)  # I = a2*V + b2
    
    # Interseção: a1*V + b1 = a2*V + b2
    # V*(a1 - a2) = b2 - b1
    # V = (b2 - b1) / (a1 - a2)
    
    if abs(a1 - a2) < 1e-10:
        return float('nan')  # Retas paralelas
    
    V_min = (b2 - b1) / (a1 - a2)
    
    return V_min

def calculate_planck_constant(V_min: float, lambda_nm: float) -> float:
    """
    Calcula a constante de Planck usando a relação do efeito fotoelétrico:
    e*V = h*f = h*c/λ  =>  h = (e*V*λ)/c
    
    Parâmetros:
    - V_min: tensão mínima em mV
    - lambda_nm: comprimento de onda em nm
    
    Retorna:
    - h: constante de Planck em J·s
    """
    # Converte V_min de mV para V
    V_min_volts = V_min / 1000.0
    
    # Converte λ de nm para m
    lambda_m = lambda_nm * 1e-9
    
    # Calcula h = (e * V_min * λ) / c
    h = (E_CHARGE * V_min_volts * lambda_m) / C_LIGHT
    
    return h

def format_uncertainty(mean: float, std: float, scale: float = 1e34) -> Tuple[str, str]:
    """
    Formata média e incerteza com 1 algarismo significativo na incerteza.
    Retorna (mean_str, std_str) já escalados.
    
    Exemplo: mean=6.626e-34, std=0.095e-34, scale=1e34
    -> std arredondado para 0.1, mean para 6.6
    """
    if std == 0 or np.isnan(std):
        mean_scaled = mean * scale
        return f"{mean_scaled:.2f}", "0.0"
    
    # Escala valores
    std_scaled = std * scale
    mean_scaled = mean * scale
    
    # Encontra ordem de magnitude da incerteza
    if std_scaled > 0:
        magnitude = 10 ** np.floor(np.log10(std_scaled))
    else:
        magnitude = 1
    
    # Arredonda incerteza para 1 algarismo significativo
    std_rounded = np.round(std_scaled / magnitude) * magnitude
    
    # Determina casas decimais baseado na magnitude
    if magnitude >= 1:
        decimals = 0
    else:
        decimals = int(-np.floor(np.log10(magnitude)))
    
    # Arredonda média para a mesma precisão
    mean_rounded = np.round(mean_scaled / magnitude) * magnitude
    
    # Formata strings
    mean_str = f"{mean_rounded:.{decimals}f}"
    std_str = f"{std_rounded:.{decimals}f}"
    
    return mean_str, std_str

def format_lambda_legend(label: str, lambdas: List[float]) -> str:
    """
    Formata a legenda com o comprimento de onda.
    Ex.: 'azul1 (λ = 466) nm' ou 'branco2 (λ1 = 373, λ2 = 448, λ3 = 560) nm'
    """
    if not lambdas:
        return label
    
    if len(lambdas) == 1:
        return f"{label} (λ = {int(round(lambdas[0]))}) nm"
    else:
        lambda_parts = [f"λ{i+1} = {int(round(lam))}" for i, lam in enumerate(lambdas)]
        return f"{label} ({', '.join(lambda_parts)}) nm"

def set_common_axes(ax) -> None:
    """Define eixos e grid comuns."""
    ax.set_xlabel("Tensão (mV)", fontsize=11)
    ax.set_ylabel("Corrente (mA)", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.25)

def generate_planck_table(
    vi_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
    plots_dir: Path,
    color_name: str,
    lambdas_dict: Dict[str, List[float]]
) -> None:
    """
    Gera tabela com análise Piecewise Linear para uma cor.
    Calcula V_min e h para cada laser (todos os lambdas).
    Salva como imagem PNG.
    """
    # Seleciona labels desta cor
    results = []
    all_h_values = []  # Para calcular estatísticas
    
    for label, (V, I) in vi_data.items():
        cor, _ = split_color_and_index(label)
        
        # Lógica especial para azul0 e brancos
        if color_name == "branco":
            # Tabela branco: inclui brancos E azul0
            if cor not in ["branco", "azul"]:
                continue
            if cor == "azul" and label != "azul0":
                continue
        elif color_name == "azul":
            # Tabela azul: exclui azul0
            if cor != "azul" or label == "azul0":
                continue
        else:
            # Outras cores: comportamento normal
            if cor != color_name:
                continue
            # Para vermelho, exclui vermelho8
            if color_name == "vermelho" and label == "vermelho8":
                continue
        
        # Calcula V_min
        V_min = calculate_v_min_piecewise(V, I)
        
        if np.isnan(V_min):
            continue
        
        # Pega lambdas para este label
        lambdas = lambdas_dict.get(label, [])
        if not lambdas:
            continue
        
        # Calcula h para TODOS os lambdas
        h_values = []
        for lambda_nm in lambdas:
            h = calculate_planck_constant(V_min, lambda_nm)
            h_values.append(h)
            all_h_values.append(h)  # Para estatísticas globais
        
        results.append((label, lambdas, V_min, h_values))
    
    if not results:
        return
    
    # Cria figura para a tabela (muito compacta)
    fig, ax = plt.subplots(figsize=(10, len(results) * 0.4 + 1.2))
    ax.axis('tight')
    ax.axis('off')
    
    # Título
    title_text = f"Análise Piecewise Linear - {color_name.capitalize()}"
    ax.set_title(title_text, fontsize=14, weight='bold', pad=5)
    
    # Define separador (| para branco, vírgula para outros)
    separator = " | " if color_name == "branco" else ", "
    
    # Prepara dados da tabela
    table_data = []
    for label, lambdas, V_min, h_values in results:
        # Formata lambdas
        lambda_str = separator.join([f"{lam:.2f}" for lam in lambdas])
        
        # Formata h values (convertidos para ×10⁻³⁴)
        h_str = separator.join([f"{h*1e34:.2f}" for h in h_values])
        
        # Converte V_min de mV para V (divide por 1000)
        V_min_V = V_min / 1000.0
        
        table_data.append([
            label,
            lambda_str,
            f"{V_min_V:.2f}",
            h_str
        ])
    
    # Cria tabela com unidades no cabeçalho
    table = ax.table(
        cellText=table_data,
        colLabels=["Label", "λ (nm)", "V_mínimo (V)", "h (×10⁻³⁴ J·s)"],
        cellLoc='center',
        loc='center',
        colWidths=[0.2, 0.25, 0.25, 0.3]
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.0)
    
    # Estiliza cabeçalho
    for i in range(4):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white', fontsize=10)
    
    # Calcula média e desvio padrão de todos os h
    h_mean = np.mean(all_h_values)
    h_std = np.std(all_h_values, ddof=1) if len(all_h_values) > 1 else 0.0
    
    # Formata com incerteza adequada
    mean_str, std_str = format_uncertainty(h_mean, h_std, scale=1e34)
    
    # Calcula δH/σ
    if h_std > 0:
        delta_h_sigma = abs(h_mean - H_PLANCK_REF) / h_std
    else:
        delta_h_sigma = 0.0
    
    # Adiciona h_ref, h_med e δH/σ embaixo da tabela (em itálico, fonte maior)
    h_ref_scaled = H_PLANCK_REF * 1e34
    summary_text = (
        f"$h_{{ref}} = {h_ref_scaled:.4f} \\times 10^{{-34}}$ J·s\n"
        f"$h_{{med}} = ({mean_str} \\pm {std_str}) \\times 10^{{-34}}$ J·s\n"
        f"$\\Delta h / \\sigma = {delta_h_sigma:.2f}$"
    )
    ax.text(0.5, -0.08, summary_text, ha='center', fontsize=13, 
            style='italic', verticalalignment='top', transform=ax.transAxes)
    
    # Salva
    out_path = plots_dir / f"tabela_{color_name}.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

def generate_planck_table_comparison(
    vi_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
    plots_dir: Path,
    labels: List[str],
    lambdas_dict: Dict[str, List[float]]
) -> None:
    """
    Gera tabela com análise Piecewise Linear para o gráfico comparativo.
    Calcula V_min e h para cada laser (todos os lambdas).
    """
    results = []
    all_h_values = []  # Para calcular estatísticas
    
    for label in labels:
        if label not in vi_data:
            continue
        
        V, I = vi_data[label]
        
        # Calcula V_min
        V_min = calculate_v_min_piecewise(V, I)
        
        if np.isnan(V_min):
            continue
        
        # Pega lambdas para este label
        lambdas = lambdas_dict.get(label, [])
        if not lambdas:
            continue
        
        # Calcula h para TODOS os lambdas
        h_values = []
        for lambda_nm in lambdas:
            h = calculate_planck_constant(V_min, lambda_nm)
            h_values.append(h)
            all_h_values.append(h)  # Para estatísticas globais
        
        results.append((label, lambdas, V_min, h_values))
    
    if not results:
        return
    
    # Cria figura para a tabela (muito compacta)
    fig, ax = plt.subplots(figsize=(10, len(results) * 0.4 + 1.2))
    ax.axis('tight')
    ax.axis('off')
    
    # Título
    title_text = "Análise Piecewise Linear - Comparativo"
    ax.set_title(title_text, fontsize=14, weight='bold', pad=5)
    
    # Prepara dados da tabela
    table_data = []
    for label, lambdas, V_min, h_values in results:
        # Formata lambdas (separados por vírgula)
        lambda_str = ", ".join([f"{lam:.2f}" for lam in lambdas])
        
        # Formata h values (convertidos para ×10⁻³⁴, separados por vírgula)
        h_str = ", ".join([f"{h*1e34:.2f}" for h in h_values])
        
        # Converte V_min de mV para V (divide por 1000)
        V_min_V = V_min / 1000.0
        
        table_data.append([
            label,
            lambda_str,
            f"{V_min_V:.2f}",
            h_str
        ])
    
    # Cria tabela com unidades no cabeçalho
    table = ax.table(
        cellText=table_data,
        colLabels=["Label", "λ (nm)", "V_mínimo (V)", "h (×10⁻³⁴ J·s)"],
        cellLoc='center',
        loc='center',
        colWidths=[0.2, 0.25, 0.25, 0.3]
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.0)
    
    # Estiliza cabeçalho
    for i in range(4):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white', fontsize=10)
    
    # Calcula média e desvio padrão de todos os h
    h_mean = np.mean(all_h_values)
    h_std = np.std(all_h_values, ddof=1) if len(all_h_values) > 1 else 0.0
    
    # Formata com incerteza adequada
    mean_str, std_str = format_uncertainty(h_mean, h_std, scale=1e34)
    
    # Calcula δH/σ
    if h_std > 0:
        delta_h_sigma = abs(h_mean - H_PLANCK_REF) / h_std
    else:
        delta_h_sigma = 0.0
    
    # Adiciona h_ref, h_med e δH/σ embaixo da tabela (em itálico, fonte maior)
    h_ref_scaled = H_PLANCK_REF * 1e34
    summary_text = (
        f"$h_{{ref}} = {h_ref_scaled:.4f} \\times 10^{{-34}}$ J·s\n"
        f"$h_{{med}} = ({mean_str} \\pm {std_str}) \\times 10^{{-34}}$ J·s\n"
        f"$\\Delta h / \\sigma = {delta_h_sigma:.2f}$"
    )
    ax.text(0.5, -0.08, summary_text, ha='center', fontsize=13, 
            style='italic', verticalalignment='top', transform=ax.transAxes)
    
    # Salva
    out_path = plots_dir / "tabela_comparativo.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

def generate_final_table(
    vi_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
    plots_dir: Path,
    lambdas_dict: Dict[str, List[float]]
) -> None:
    """
    Gera tabela final com todos os h calculados em layout duplo (2 subtabelas lado a lado).
    """
    results = []
    all_h_values = []
    
    for label, (V, I) in sorted(vi_data.items()):
        cor, _ = split_color_and_index(label)
        
        # Pula vermelho8
        if label == "vermelho8":
            continue
        
        # Calcula V_min
        V_min = calculate_v_min_piecewise(V, I)
        
        if np.isnan(V_min):
            continue
        
        # Pega lambdas
        lambdas = lambdas_dict.get(label, [])
        if not lambdas:
            continue
        
        # Calcula h para TODOS os lambdas
        h_values = []
        for lambda_nm in lambdas:
            h = calculate_planck_constant(V_min, lambda_nm)
            h_values.append(h)
            all_h_values.append(h)
        
        results.append((label, h_values))
    
    if not results:
        return
    
    # Divide em três partes
    n = len(results)
    third = (n + 2) // 3
    results_left = results[:third]
    results_center = results[third:2*third]
    results_right = results[2*third:]
    
    max_rows = max(len(results_left), len(results_center), len(results_right))
    
    # Cria figura com três subtabelas (mais compacta)
    fig = plt.figure(figsize=(8, max_rows * 0.35 + 1.5))
    
    # Título geral
    fig.suptitle('Tabela Final - Todos os Valores de h', fontsize=14, weight='bold', y=0.97)
    
    # Subtabela esquerda
    ax_left = fig.add_subplot(1, 3, 1)
    ax_left.axis('tight')
    ax_left.axis('off')
    
    table_data_left = []
    for label, h_values in results_left:
        h_str = ", ".join([f"{h*1e34:.2f}" for h in h_values])
        table_data_left.append([label, h_str])
    
    table_left = ax_left.table(
        cellText=table_data_left,
        colLabels=["Label", "h (×10⁻³⁴ J·s)"],
        cellLoc='center',
        loc='center',
        colWidths=[0.5, 0.5]
    )
    table_left.auto_set_font_size(False)
    table_left.set_fontsize(9)
    table_left.scale(1, 1.6)
    
    for i in range(2):
        table_left[(0, i)].set_facecolor('#4472C4')
        table_left[(0, i)].set_text_props(weight='bold', color='white', fontsize=10)
    
    # Subtabela centro
    ax_center = fig.add_subplot(1, 3, 2)
    ax_center.axis('tight')
    ax_center.axis('off')
    
    table_data_center = []
    for label, h_values in results_center:
        h_str = ", ".join([f"{h*1e34:.2f}" for h in h_values])
        table_data_center.append([label, h_str])
    
    # Preenche com linhas vazias
    while len(table_data_center) < len(table_data_left):
        table_data_center.append(["", ""])
    
    table_center = ax_center.table(
        cellText=table_data_center,
        colLabels=["Label", "h (×10⁻³⁴ J·s)"],
        cellLoc='center',
        loc='center',
        colWidths=[0.5, 0.5]
    )
    table_center.auto_set_font_size(False)
    table_center.set_fontsize(9)
    table_center.scale(1, 1.6)
    
    for i in range(2):
        table_center[(0, i)].set_facecolor('#4472C4')
        table_center[(0, i)].set_text_props(weight='bold', color='white', fontsize=10)
    
    # Subtabela direita
    ax_right = fig.add_subplot(1, 3, 3)
    ax_right.axis('tight')
    ax_right.axis('off')
    
    table_data_right = []
    for label, h_values in results_right:
        h_str = ", ".join([f"{h*1e34:.2f}" for h in h_values])
        table_data_right.append([label, h_str])
    
    # Preenche com linhas vazias
    while len(table_data_right) < len(table_data_left):
        table_data_right.append(["", ""])
    
    table_right = ax_right.table(
        cellText=table_data_right,
        colLabels=["Label", "h (×10⁻³⁴ J·s)"],
        cellLoc='center',
        loc='center',
        colWidths=[0.5, 0.5]
    )
    table_right.auto_set_font_size(False)
    table_right.set_fontsize(9)
    table_right.scale(1, 1.6)
    
    for i in range(2):
        table_right[(0, i)].set_facecolor('#4472C4')
        table_right[(0, i)].set_text_props(weight='bold', color='white', fontsize=10)
    
    # Calcula estatísticas
    h_mean = np.mean(all_h_values)
    h_std = np.std(all_h_values, ddof=1) if len(all_h_values) > 1 else 0.0
    
    # Formata
    mean_str, std_str = format_uncertainty(h_mean, h_std, scale=1e34)
    
    # Calcula Δh/σ
    if h_std > 0:
        delta_h_sigma = abs(h_mean - H_PLANCK_REF) / h_std
    else:
        delta_h_sigma = 0.0
    
    # Texto embaixo
    h_ref_scaled = H_PLANCK_REF * 1e34
    summary_text = (
        f"$h_{{ref}} = {h_ref_scaled:.4f} \\times 10^{{-34}}$ J·s\n"
        f"$h_{{med}} = ({mean_str} \\pm {std_str}) \\times 10^{{-34}}$ J·s\n"
        f"$\\Delta h / \\sigma = {delta_h_sigma:.2f}$"
    )
    fig.text(0.5, 0.02, summary_text, ha='center', fontsize=13, 
             style='italic', verticalalignment='bottom')
    
    # Salva
    out_path = plots_dir / "tabela_final.png"
    fig.tight_layout(rect=[0, 0.12, 1, 0.96])
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

def plot_piecewise_demo(
    vi_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
    plots_dir: Path
) -> None:
    """
    Gera gráfico demonstrativo do método Piecewise Linear usando vermelho8.
    """
    label = "vermelho8"
    
    if label not in vi_data:
        return
    
    V, I = vi_data[label]
    
    if len(V) < 8:
        return
    
    # Calcula V_min e as retas
    V1 = V[:4]
    I1 = I[:4]
    a1, b1 = linear_regression(V1, I1)
    
    V2 = V[-4:]
    I2 = I[-4:]
    a2, b2 = linear_regression(V2, I2)
    
    if abs(a1 - a2) < 1e-10:
        return
    
    V_min = (b2 - b1) / (a1 - a2)
    I_min = a1 * V_min + b1
    
    # Cria figura
    fig, ax = plt.subplots(figsize=(9, 6))
    
    # Plota curva original (AZUL)
    ax.plot(V, I, 'o-', label='Dados experimentais', markersize=5, linewidth=1.5, color='blue')
    
    # Plota primeira reta (PRETA SÓLIDA)
    V_range1 = np.array([V.min(), V_min])
    I_range1 = a1 * V_range1 + b1
    ax.plot(V_range1, I_range1, '-', label='Reta 1 (primeiros 4 pontos)', linewidth=2, color='black')
    
    # Plota segunda reta (PRETA SÓLIDA)
    V_range2 = np.array([V_min, V.max()])
    I_range2 = a2 * V_range2 + b2
    ax.plot(V_range2, I_range2, '-', label='Reta 2 (últimos 4 pontos)', linewidth=2, color='black')
    
    # Marca ponto de interseção
    ax.plot(V_min, I_min, 'ro', markersize=10, zorder=5)
    
    # Configurações (ylim começa em 0)
    ax.set_xlabel('Tensão (mV)', fontsize=12)
    ax.set_ylabel('Corrente (mA)', fontsize=12)
    ax.set_title('Piecewise Linear - Laser (vermelho8)', fontsize=14, weight='bold')
    ax.set_ylim(bottom=0)  # Começa do zero
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # Linha vertical tracejada do eixo X até o ponto (DEPOIS de setar ylim)
    ax.plot([V_min, V_min], [0, I_min], 'r--', linewidth=1.5, alpha=0.7)
    
    # Texto V_mínimo (EMBAIXO do ponto)
    y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
    ax.text(V_min, I_min - y_range * 0.05, 
            '$V_{min}$', fontsize=14, weight='bold', color='red', 
            ha='center', va='top')
    
    ax.legend(loc='best', frameon=True)
    
    # Salva
    out_path = plots_dir / "piecewise_linear_demo.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

def plot_by_color(
    vi_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
    plots_dir: Path,
    color_name: str,
    lambdas_dict: Dict[str, List[float]]
) -> None:
    """
    Gera um gráfico I vs V para uma cor específica (azul, branco, laranja, verde, vermelho),
    contendo todas as medidas daquela cor.
    """
    # Seleciona labels desta cor
    series = []
    for label, (V, I) in vi_data.items():
        cor, _ = split_color_and_index(label)
        if cor == color_name:
            # Para vermelho, exclui vermelho8 do gráfico filtrado
            if color_name == "vermelho" and label == "vermelho8":
                continue
            series.append((label, V, I))
    
    if not series:
        return  # se não houver, não gera
    
    # Encontra o menor valor máximo de corrente entre todas as curvas
    max_currents = [I[-1] if len(I) > 0 else 0 for _, _, I in series]
    I_limit = min(max_currents) if max_currents else float('inf')
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Cada curva com uma cor diferente
    for label, V, I in series:
        # Filtra dados para mostrar apenas até o limite de corrente
        mask = I <= I_limit
        V_filtered = V[mask]
        I_filtered = I[mask]
        
        # Adiciona λ(s) na legenda
        lambdas = lambdas_dict.get(label, [])
        label_text = format_lambda_legend(label, lambdas)
        ax.plot(V_filtered, I_filtered, label=label_text, marker='o', markersize=4, alpha=0.9, linewidth=1.6)
    
    title = f"Curva I vs V — Cor: {color_name.capitalize()}"
    ax.set_title(title)
    set_common_axes(ax)
    ax.legend(title="Laser", frameon=False, loc='best')
    out_path = plots_dir / f"I_vs_V_{color_name}.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

def plot_vermelho_unfiltered(
    vi_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
    plots_dir: Path,
    lambdas_dict: Dict[str, List[float]]
) -> None:
    """
    Gera um gráfico I vs V para vermelho SEM FILTRO.
    Mostra todos os dados de todos os vermelhos, incluindo vermelho8.
    """
    # Seleciona todos os labels vermelho
    series = []
    for label, (V, I) in vi_data.items():
        cor, _ = split_color_and_index(label)
        if cor == "vermelho":
            series.append((label, V, I))
    
    if not series:
        return  # se não houver, não gera
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Cada curva com uma cor diferente - SEM FILTRO
    for label, V, I in series:
        # Adiciona λ(s) na legenda
        lambdas = lambdas_dict.get(label, [])
        label_text = format_lambda_legend(label, lambdas)
        ax.plot(V, I, label=label_text, marker='o', markersize=4, alpha=0.9, linewidth=1.6)
    
    title = "Curva I vs V — Cor: Vermelho (Sem Filtro)"
    ax.set_title(title)
    set_common_axes(ax)
    ax.legend(title="Laser", frameon=False, loc='best')
    out_path = plots_dir / "I_vs_V_vermelho_sem_filtro.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

def plot_all_lasers(
    vi_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
    plots_dir: Path
) -> None:
    """
    Gera um gráfico I vs V com TODOS os lasers.
    Legenda do lado de fora (direita) sem comprimentos de onda.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plota todos os lasers
    for label in sorted(vi_data.keys()):
        V, I = vi_data[label]
        cor, _ = split_color_and_index(label)
        
        # Usa apenas o label, sem comprimento de onda
        ax.plot(V, I, label=label, marker='o', markersize=3, 
                linewidth=1.5, alpha=0.85, color=COLOR_MAP.get(cor, "tab:gray"))
    
    ax.set_title("Curva I vs V — Todos os LEDs")
    set_common_axes(ax)
    
    # Legenda do lado de fora, à direita
    ax.legend(title="LED", frameon=True, loc='center left', bbox_to_anchor=(1, 0.5))
    
    out_path = plots_dir / "I_vs_V.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

def plot_comparison(
    vi_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
    plots_dir: Path,
    labels: List[str],
    lambdas_dict: Dict[str, List[float]]
) -> None:
    """
    Gera o gráfico comparativo I vs V com os labels informados.
    Usa COLOR_MAP para colorir por cor do laser.
    """
    # Coleta dados válidos e encontra o menor valor máximo de corrente
    valid_series = []
    for label in labels:
        if label in vi_data:
            V, I = vi_data[label]
            valid_series.append((label, V, I))
    
    if not valid_series:
        return  # Nada para plotar
    
    # Encontra o menor valor máximo de corrente entre todas as curvas
    max_currents = [I[-1] if len(I) > 0 else 0 for _, _, I in valid_series]
    I_limit = min(max_currents) if max_currents else float('inf')
    
    fig, ax = plt.subplots(figsize=(9, 5.5))
    
    for label, V, I in valid_series:
        # Filtra dados para mostrar apenas até o limite de corrente
        mask = I <= I_limit
        V_filtered = V[mask]
        I_filtered = I[mask]
        
        cor, _ = split_color_and_index(label)
        
        # Adiciona λ(s) na legenda
        lambdas = lambdas_dict.get(label, [])
        label_text = format_lambda_legend(label, lambdas)
        
        ax.plot(V_filtered, I_filtered, label=label_text, marker='o', markersize=4, 
                linewidth=1.8, alpha=0.95, color=COLOR_MAP.get(cor, "tab:gray"))
    
    ax.set_title("Comparativo — Curvas I vs V (seleção configurável)")
    set_common_axes(ax)
    ax.legend(title="Laser", frameon=False, ncol=1, loc='best')
    out_path = plots_dir / "I_vs_V_comparativo.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

# =========================
# PIPELINE
# =========================
def main() -> None:
    ensure_dirs()
    
    # Etapa 1 — leitura dos dados V-I
    vi_data = read_all_vi_data(DATA_DIR)
    
    if not vi_data:
        print("Erro: Nenhum dado V-I encontrado!")
        return
    
    # Etapa 2 — importa comprimentos de onda
    lambdas_dict = read_lambda_mapping(LAMBDA_CSV)
    
    # Etapa 3 — gráficos
    for cor in ["azul", "branco", "laranja", "verde", "vermelho"]:
        plot_by_color(vi_data, PLOTS_DIR, cor, lambdas_dict)
    
    # Gráfico especial: vermelho sem filtro (com todos os dados)
    plot_vermelho_unfiltered(vi_data, PLOTS_DIR, lambdas_dict)
    
    # Gráfico com todos os lasers
    plot_all_lasers(vi_data, PLOTS_DIR)
    
    plot_comparison(vi_data, PLOTS_DIR, COMPARISON_LABELS, lambdas_dict)
    
    # Etapa 4 — análise Piecewise Linear e tabelas
    for cor in ["azul", "branco", "laranja", "verde", "vermelho"]:
        generate_planck_table(vi_data, PLOTS_DIR, cor, lambdas_dict)
    
    generate_planck_table_comparison(vi_data, PLOTS_DIR, COMPARISON_LABELS, lambdas_dict)
    
    # Tabela final com todos os h
    generate_final_table(vi_data, PLOTS_DIR, lambdas_dict)
    
    # Gráfico demonstrativo do piecewise linear
    plot_piecewise_demo(vi_data, PLOTS_DIR)
    
    # Prints finais
    print("[1/3] Dados V-I lidos e processados.")
    print(f"[2/3] Gráficos I vs V gerados em: {PLOTS_DIR}")
    print(f"[3/3] Tabelas de análise Piecewise Linear geradas em: {PLOTS_DIR}")

if __name__ == "__main__":
    main()