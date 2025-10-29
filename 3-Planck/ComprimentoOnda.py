"""
comprimento_de_onda.py

Lê espectros (λ [nm], intensidade) de TXT em 'dados/comprimento de onda/',
estima λ por média do x nos 10 maiores y, salva 'dados/lambda-outputs.csv'
e gera 6 gráficos normalizados (5 por cor + 1 comparativo configurável).
"""

from __future__ import annotations
from pathlib import Path
import re
import math
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# CONFIGURAÇÃO DE PATHS
# =========================
# Ajuste estes caminhos conforme a sua estrutura local.
PROJECT_DIR = Path.cwd() / "3-Planck"
DATA_DIR = PROJECT_DIR / "Dados" / "ComprimentoOnda"   # pasta com os .txt
CSV_OUT = PROJECT_DIR / "Dados" / "Lambda_Outputs.csv"     # saída exigida
PLOTS_DIR = PROJECT_DIR / "Graficos"             # onde salvar os .png

# =========================
# PARÂMETROS
# =========================
# Precisão desejada para λ (em nm). O padrão aqui é 0,01 nm, como solicitado.
LAMBDA_PRECISION_DECIMALS = 2  # 0,01 nm
TOP_K = 10  # número de maiores intensidades para a média do λ

# Rótulos (labels) a compor o 6º gráfico comparativo (edite à vontade):
COMPARISON_LABELS = ["azul1", "laranja1", "verde1", "vermelho1"]

# =========================
# LAMBDAS MANUAIS (nm)
# =========================
# Defina manualmente os comprimentos de onda para brancos e azul0
# Use o script interativo (medidor_lambdas.py) para medir as coordenadas
MANUAL_LAMBDAS = {
    "azul0": [399.67, 446.03, 556.25],  # Medido manualmente
    "branco1": [448.09, 566.55],  # Medido manualmente
    "branco2": [372.89, 448.09, 560.37],  # Medido manualmente
    "branco3": [449.12, 574.79],  # Medido manualmente
    "branco4": [454.27, 554.19],  # Medido manualmente
    "branco5": [371.86, 448.09, 556.25],  # Medido manualmente
}

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
    (PROJECT_DIR / "Dados").mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

def parse_label(path: Path) -> str:
    """
    Extrai o label a partir do nome do arquivo.
    Ex.: 'azul0.txt' -> 'azul0'
    """
    return path.stem

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

def try_read_two_column_txt(path: Path) -> pd.DataFrame:
    """
    Lê um TXT de duas colunas numéricas (λ, intensidade).
    Tenta formatos comuns:
      1) sep=';' e decimal=','  (muito comum em PT-BR)
      2) sep=whitespace e decimal=',' (colunas por espaço/tab, decimais com vírgula)
      3) sep=',' e decimal='.'  (CSV simples em EN)
    Retorna DataFrame com colunas [0, 1] (float).
    """
    # 1) ponto e vírgula + vírgula decimal
    try:
        df = pd.read_csv(path, header=None, sep=";", decimal=",", engine="python")
        if df.shape[1] >= 2:
            df = df.iloc[:, :2].astype(float)
            return df
    except Exception:
        pass
    # 2) whitespace + vírgula decimal
    try:
        df = pd.read_csv(path, header=None, sep=r"\s+", decimal=",", engine="python")
        if df.shape[1] >= 2:
            df = df.iloc[:, :2].astype(float)
            return df
    except Exception:
        pass
    # 3) vírgula separador + ponto decimal
    try:
        df = pd.read_csv(path, header=None, sep=",", decimal=".", engine="python")
        if df.shape[1] >= 2:
            df = df.iloc[:, :2].astype(float)
            return df
    except Exception:
        pass

    # Caso alguma variação exótica: leitura manual simples (foco em robustez mínima)
    xs, ys = [], []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Tenta separar por ; , ou whitespace
            if ";" in line:
                parts = line.split(";")
            elif "," in line and " " not in line and "\t" not in line:
                # Se só há vírgulas, assume separador vírgula e decimal ponto
                parts = line.split(",")
            else:
                parts = re.split(r"\s+", line)

            if len(parts) < 2:
                continue

            a, b = parts[0].strip(), parts[1].strip()
            # Converte vírgula decimal para ponto, se houver
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

def read_all_spectra(data_dir: Path) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Lê todos os arquivos .txt e devolve um dicionário:
      { label: (x_nm, y) }
    """
    spectra: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for path in sorted(data_dir.glob("*.txt")):
        label = parse_label(path)
        df = try_read_two_column_txt(path)
        x = df.iloc[:, 0].to_numpy(dtype=float)
        y = df.iloc[:, 1].to_numpy(dtype=float)
        # Ordena por x caso venha bagunçado
        idx = np.argsort(x)
        x, y = x[idx], y[idx]
        spectra[label] = (x, y)
    return spectra

def find_peaks_iterative(x: np.ndarray, y: np.ndarray, num_peaks: int, k: int = 30, deriv_window: int = 5) -> List[float]:
    """
    Encontra múltiplos picos iterativamente:
    1. Encontra o máximo global
    2. Calcula lambda com média dos k pontos ao redor (padrão: 30 pontos)
    3. Remove a região do pico usando análise de derivada
    4. Repete para o próximo pico
    
    Retorna lista de lambdas em ordem crescente.
    """
    # Cria cópia dos dados para manipular
    x_work = x.copy()
    y_work = y.copy()
    lambdas = []
    
    for peak_num in range(num_peaks):
        if len(y_work) < 2 * k:
            break  # Não há dados suficientes
        
        # 1. Encontra o máximo global
        max_idx = np.argmax(y_work)
        
        # 2. Calcula lambda com média ponderada dos k pontos ao redor do máximo
        start = max(0, max_idx - k // 2)
        end = min(len(x_work), max_idx + k // 2 + 1)
        x_region = x_work[start:end]
        y_region = y_work[start:end]
        
        if len(y_region) > 0 and np.sum(y_region) > 0:
            lam = np.average(x_region, weights=y_region)
            lambdas.append(float(lam))
        
        # 3. Remove a região do pico
        # Marca pontos para remover
        to_remove = np.zeros(len(y_work), dtype=bool)
        to_remove[max_idx] = True
        
        # Remove pontos ANTES do pico onde derivada é positiva (subindo)
        for i in range(max_idx - 1, deriv_window - 1, -1):
            if i < deriv_window:
                break
            # Calcula derivada média em uma janela
            window = y_work[i - deriv_window:i + 1]
            if len(window) >= 2:
                deriv = np.mean(np.diff(window))
                if deriv > 0:  # Ainda está subindo
                    to_remove[i] = True
                else:
                    break  # Parou de subir, acabou a região do pico
        
        # Remove pontos DEPOIS do pico onde derivada é negativa (descendo)
        for i in range(max_idx + 1, len(y_work) - deriv_window):
            if i + deriv_window >= len(y_work):
                break
            # Calcula derivada média em uma janela
            window = y_work[i:i + deriv_window + 1]
            if len(window) >= 2:
                deriv = np.mean(np.diff(window))
                if deriv < 0:  # Ainda está descendo
                    to_remove[i] = True
                else:
                    break  # Começou a subir, novo pico à vista
        
        # Remove os pontos marcados
        keep_mask = ~to_remove
        x_work = x_work[keep_mask]
        y_work = y_work[keep_mask]
    
    # Retorna em ordem crescente
    lambdas.sort()
    return lambdas

def estimate_lambda_from_topk(x: np.ndarray, y: np.ndarray, k: int = TOP_K) -> float:
    """
    Média do x nos k maiores valores de y.
    Se houver menos de k pontos, usa todos.
    """
    n = len(y)
    if n == 0:
        return float("nan")
    k = min(k, n)
    # índices dos maiores k y (sem ordenar todos)
    idx_top = np.argpartition(y, -k)[-k:]
    x_top = x[idx_top]
    return float(np.mean(x_top))

def estimate_multiple_lambdas(x: np.ndarray, y: np.ndarray, label: str, k: int = TOP_K) -> List[float]:
    """
    Estima múltiplos comprimentos de onda para labels específicos (brancos e azul0).
    Usa valores manuais do dicionário MANUAL_LAMBDAS.
    Retorna lista de lambdas em ordem crescente.
    """
    # Verifica se o label tem lambdas manuais definidos
    if label in MANUAL_LAMBDAS:
        # Retorna os valores manuais já em ordem crescente
        lambdas = sorted([float(lam) for lam in MANUAL_LAMBDAS[label]])
        return lambdas
    
    # Comportamento padrão para outros labels: um único lambda
    return [estimate_lambda_from_topk(x, y, k)]

def round_lambda(value_nm: float, decimals: int = LAMBDA_PRECISION_DECIMALS) -> float:
    """Arredonda λ à precisão desejada (ex.: 2 casas decimais -> 0,01 nm)."""
    if math.isnan(value_nm):
        return value_nm
    return round(value_nm, decimals)

def write_lambda_csv(results: List[Tuple[str, List[float]]], out_csv: Path) -> None:
    """
    Escreve CSV 'label,lambda' (ponto como separador decimal).
    Quando há múltiplos lambdas, separa por '/'.
    Sobrescreve o arquivo para refletir o estado atual.
    """
    # Converte lista de lambdas para string com separador '/'
    formatted_results = []
    for label, lambdas in results:
        if isinstance(lambdas, list):
            lambda_str = "/".join([str(lam) for lam in lambdas])
        else:
            lambda_str = str(lambdas)
        formatted_results.append((label, lambda_str))
    
    df = pd.DataFrame(formatted_results, columns=["label", "lambda"])
    # Garante ponto como decimal; to_csv padrão já usa ponto
    df.to_csv(out_csv, index=False, encoding="utf-8")

def normalize(y: np.ndarray) -> np.ndarray:
    """Normaliza para [0, 1] pelo pico (máximo)."""
    m = np.max(y) if len(y) else 1.0
    return y / m if m != 0 else y

def set_common_axes(ax) -> None:
    ax.set_xlabel("Comprimento de onda (nm)")
    ax.set_ylabel("Intensidade normalizada (a.u.)")
    ax.grid(True, linestyle="--", alpha=0.25)

def plot_by_color(
    spectra: Dict[str, Tuple[np.ndarray, np.ndarray]],
    plots_dir: Path,
    color_name: str,
    lambdas: Dict[str, List[float]]
) -> None:
    """
    Gera um gráfico para uma cor específica (azul, branco, laranja, verde, vermelho),
    contendo todas as medidas daquela cor (ex.: azul0, azul1, ...), normalizadas.
    Para o gráfico branco, inclui também azul0.
    """
    # Seleciona labels desta cor
    series = []
    for label, (x, y) in spectra.items():
        cor, _ = split_color_and_index(label)
        if cor == color_name:
            series.append((label, x, normalize(y)))
        # Adiciona azul0 ao gráfico branco
        elif color_name == "branco" and label == "azul0":
            series.append((label, x, normalize(y)))
    
    if not series:
        return  # se não houver, não gera

    fig, ax = plt.subplots(figsize=(8, 5))
    # Cada curva com uma cor diferente
    for label, x, y in series:
        # Adiciona λ(s) na legenda (arredondado para unidade)
        lambda_vals = lambdas.get(label, [])
        if isinstance(lambda_vals, list) and len(lambda_vals) > 1:
            # Múltiplos lambdas: (λ1 = 100, λ2 = 150) nm
            lambda_parts = [f"λ{i+1} = {int(round(lam))}" for i, lam in enumerate(lambda_vals)]
            label_text = f"{label} ({', '.join(lambda_parts)}) nm"
        elif isinstance(lambda_vals, list) and len(lambda_vals) == 1:
            # Um único lambda em lista
            label_text = f"{label} (λ = {int(round(lambda_vals[0]))}) nm"
        elif not isinstance(lambda_vals, list) and not math.isnan(lambda_vals):
            # Lambda como float direto
            label_text = f"{label} (λ = {int(round(lambda_vals))}) nm"
        else:
            label_text = label
        ax.plot(x, y, label=label_text, alpha=0.9, linewidth=1.6)

    title = f"Distribuição espectral normalizada — Cor: {color_name.capitalize()}"
    ax.set_title(title)
    set_common_axes(ax)
    ax.legend(title="Laser", frameon=False)
    out_path = plots_dir / f"espectro_{color_name}.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

def plot_comparison(
    spectra: Dict[str, Tuple[np.ndarray, np.ndarray]],
    plots_dir: Path,
    labels: List[str],
    lambdas: Dict[str, List[float]]
) -> None:
    """
    Gera o 6º gráfico comparativo com os labels informados (normalizados).
    Usa COLOR_MAP para colorir por cor do laser.
    """
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for label in labels:
        if label not in spectra:
            # Se algum label não existe, simplesmente ignora.
            continue
        x, y = spectra[label]
        y = normalize(y)
        cor, _ = split_color_and_index(label)
        
        # Adiciona λ(s) na legenda (arredondado para unidade)
        lambda_vals = lambdas.get(label, [])
        if isinstance(lambda_vals, list) and len(lambda_vals) > 1:
            # Múltiplos lambdas: (λ1 = 100, λ2 = 150) nm
            lambda_parts = [f"λ{i+1} = {int(round(lam))}" for i, lam in enumerate(lambda_vals)]
            label_text = f"{label} ({', '.join(lambda_parts)}) nm"
        elif isinstance(lambda_vals, list) and len(lambda_vals) == 1:
            # Um único lambda em lista
            label_text = f"{label} (λ = {int(round(lambda_vals[0]))}) nm"
        elif not isinstance(lambda_vals, list) and not math.isnan(lambda_vals):
            # Lambda como float direto
            label_text = f"{label} (λ = {int(round(lambda_vals))}) nm"
        else:
            label_text = label
        
        ax.plot(x, y, label=label_text, linewidth=1.8, alpha=0.95, color=COLOR_MAP.get(cor, "tab:gray"))

    ax.set_title("Comparativo — Espectros normalizados (seleção configurável)")
    set_common_axes(ax)
    ax.legend(title="Laser", frameon=False, ncol=1, loc='upper right')
    out_path = plots_dir / "espectro_comparativo_selecao.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

# =========================
# PIPELINE
# =========================
def main() -> None:
    ensure_dirs()

    # Etapa 1 — leitura e estimação de λ
    spectra = read_all_spectra(DATA_DIR)
    results: List[Tuple[str, List[float]]] = []
    for label, (x, y) in spectra.items():
        # Usa detecção de múltiplos picos para brancos e azul0
        lambdas = estimate_multiple_lambdas(x, y, label, k=TOP_K)
        # Arredonda cada lambda
        lambdas_rounded = [round_lambda(lam, decimals=LAMBDA_PRECISION_DECIMALS) for lam in lambdas]
        results.append((label, lambdas_rounded))

    # Ordena resultados por label para estabilidade
    results.sort(key=lambda t: t[0])

    # Etapa 2 — gravação do CSV
    write_lambda_csv(results, CSV_OUT)

    # Cria dicionário de lambdas para usar nas legendas
    lambdas_dict = {label: lams for label, lams in results}

    # Etapa 3 — gráficos
    for cor in ["azul", "branco", "laranja", "verde", "vermelho"]:
        plot_by_color(spectra, PLOTS_DIR, cor, lambdas_dict)

    plot_comparison(spectra, PLOTS_DIR, COMPARISON_LABELS, lambdas_dict)

    # Prints minimalistas solicitados
    print("[1/3] Dados lidos e λ estimado.")
    print(f"[2/3] CSV atualizado em: {CSV_OUT}")
    print(f"[3/3] Gráficos gerados em: {PLOTS_DIR}")

if __name__ == "__main__":
    main()
