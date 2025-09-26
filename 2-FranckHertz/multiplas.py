import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit
import re
import io

# Configurar matplotlib para melhor visualização (mesma identidade visual do graph.py)
plt.rcParams['figure.figsize'] = (10, 8)
plt.rcParams['font.size'] = 12

# Diretórios base
base_data_dir = '/home/nico/Documentos/Lab_Avancado/2-FranckHertz/Dados/Multiplas'
base_output_dir = '/home/nico/Documentos/Lab_Avancado/2-FranckHertz/Graficos/Multiplas'

# Temperaturas e suas informações
temperaturas_info = {
    'T125': {'coef_dir': '1_8', 'temp_celsius': '125 °C', 'coef_value': 1e-8},  # 1_8 -> 1 * 10^-8
    'T145': {'coef_dir': '3_9', 'temp_celsius': '145 °C', 'coef_value': 3e-9},  # 3_9 -> 3 * 10^-9
    'T160': {'coef_dir': '3_9', 'temp_celsius': '160 °C', 'coef_value': 3e-9},  # 3_9 -> 3 * 10^-9
    'T175': {'coef_dir': '3_9', 'temp_celsius': '175 °C', 'coef_value': 3e-9},  # 3_9 -> 3 * 10^-9
    'T185': {'coef_dir': '3_9', 'temp_celsius': '185 °C', 'coef_value': 3e-9}   # 3_9 -> 3 * 10^-9
}

# Função para converter strings com vírgula para float
def convert_to_float(value):
    if isinstance(value, str):
        value = value.strip()
        if not value or value == '0':
            return 0.0
        try:
            return float(value.replace(',', '.'))
        except ValueError:
            return np.nan
    elif pd.isna(value):
        return np.nan
    return float(value)

# Função para extrair parâmetros do nome do arquivo: {V1}_{V2}_{V4}.txt
def extract_params(filename):
    match = re.match(r'(\d+(?:\.\d+)?)_(\d+(?:\.\d+)?)_(\d+(?:\.\d+)?)\.txt', filename)
    if match:
        v1, v2, v4 = map(float, match.groups())
        return v1, v2, v4
    else:
        print(f"Nome de arquivo inválido: {filename}")
        return None, None, None

# Função para determinar qual tipo de variação (V1, V2 ou V4) um arquivo pertence
def classify_file(v1, v2, v4, temp_folder):
    """
    Classifica o arquivo baseado nos valores V1, V2, V4 e temperatura
    Retorna: 'variando_V1', 'variando_V2', 'variando_V4', ou None
    """
    print(f"DEBUG classify_file: temp={temp_folder}, v1={v1}, v2={v2}, v4={v4}")
    
    if temp_folder == 'T175':
        # Regras especiais para T175
        if v2 == 6.0 and v4 == 0.5:
            print("  -> variando_V1 (T175: V2=6.0, V4=0.5)")
            return 'variando_V1'
        if v1 == 2.0 and v4 == 0.5:
            print("  -> variando_V2 (T175: V1=2.0, V4=0.5)")
            return 'variando_V2'
        if v1 == 2.0 and v2 == 6.0:
            print("  -> variando_V4 (T175: V1=2.0, V2=6.0)")
            return 'variando_V4'
    else:
        # Regras gerais para outras temperaturas - CORRIGIDO: V4 primeiro, depois V1
        if v1 == 6.0 and v2 == 6.0:
            print("  -> variando_V4 (geral: V1=6.0, V2=6.0)")
            return 'variando_V4'
        if v2 == 6.0 and v4 == 0.5:
            print("  -> variando_V1 (geral: V2=6.0, V4=0.5)")
            return 'variando_V1'
        if v1 == 6.0 and v4 == 0.5:
            print("  -> variando_V2 (geral: V1=6.0, V4=0.5)")
            return 'variando_V2'
    
    print("  -> não classificado")
    return None  # Arquivo não se encaixa em nenhuma categoria

# Função para ler e processar um arquivo .txt
def read_file_data(file_path):
    """
    Lê um arquivo .txt e retorna os dados de tensão e corrente válidos
    """
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        clean_lines = []
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            parts = re.split(r'\s+', line)
            if len(parts) >= 2:
                try:
                    convert_to_float(parts[0])
                    convert_to_float(parts[1])
                    clean_lines.append(' '.join(parts[:2]))
                except (ValueError, TypeError):
                    continue
        
        if not clean_lines:
            return None, None
        
        df = pd.read_csv(io.StringIO('\n'.join(clean_lines)), 
                       sep='\s+', header=None,
                       names=['tensao', 'corrente'])
        
        tensao = df['tensao'].apply(convert_to_float).values
        corrente = df['corrente'].apply(convert_to_float).values
        
        valid_idx = (~np.isnan(tensao) & ~np.isnan(corrente) & 
                    np.isfinite(tensao) & np.isfinite(corrente))
        
        return tensao[valid_idx], corrente[valid_idx]
        
    except Exception as e:
        print(f"Erro ao ler {file_path}: {e}")
        return None, None

# Função para gerar um gráfico específico (variando_V1, variando_V2 ou variando_V4)
def generate_graph(data_list, graph_type, temp_folder, temp_celsius, output_dir, coef_value):
    """
    Gera um gráfico para um tipo específico de variação
    """
    if not data_list:
        print(f"Nenhum dado para {graph_type} em {temp_folder}")
        return
    
    # Ordenar os dados pelo parâmetro que varia (crescente)
    if graph_type == 'variando_V1':
        data_list.sort(key=lambda x: x['v1'])
    elif graph_type == 'variando_V2':
        data_list.sort(key=lambda x: x['v2'])
    elif graph_type == 'variando_V4':
        data_list.sort(key=lambda x: x['v4'])
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    cores = ['green', 'darkgreen', 'olive', 'orange', 'orangered', 'red', 'darkred', 'blue', 'darkblue']
    
    for i, data in enumerate(data_list):
        tensao_valid = data['tensao']
        corrente_valid = data['corrente']
        v1, v2, v4 = data['v1'], data['v2'], data['v4']
        
        color = cores[i % len(cores)]
        
        # Aplicar conversão para nanoampères: dado * coef * 1e9
        corrente_scaled = corrente_valid * coef_value * 1e9
        
        # Ordenar por tensão para conexão correta
        sort_idx = np.argsort(tensao_valid)
        tensao_sorted = tensao_valid[sort_idx]
        corrente_sorted = corrente_scaled[sort_idx]
        
        # Determinar label baseado no tipo de variação (mostrar apenas o que varia)
        if graph_type == 'variando_V1':
            label = f'V1={v1}V'
        elif graph_type == 'variando_V2':
            label = f'V2={v2}V'
        elif graph_type == 'variando_V4':
            label = f'V4={v4}V'
        
        # Plot dos dados conectados por linhas simples
        ax.plot(tensao_sorted, corrente_sorted, 
               color=color, linewidth=2, alpha=0.8,
               label=label,
               marker='')
        
        print(f"  {data['file']}: {label}, {len(tensao_sorted)} pontos")
    
    # Configurar o gráfico
    ax.set_xlabel('Tensão V3 (V)', fontsize=14)
    ax.set_ylabel('Corrente (nA)', fontsize=14)
    
    # Título específico para cada tipo de variação com valores fixos (quebra de linha)
    var_name = graph_type.split('_')[1]  # Extrai V1, V2 ou V4
    
    # Determinar valores fixos baseado no tipo e temperatura
    if temp_folder == 'T175':
        if graph_type == 'variando_V1':
            fixed_values = "V2 = 6.0 V e V4 = 0.5 V fixos"
        elif graph_type == 'variando_V2':
            fixed_values = "V1 = 2.0 V e V4 = 0.5 V fixos"
        elif graph_type == 'variando_V4':
            fixed_values = "V1 = 2.0 V e V2 = 6.0 V fixos"
    else:
        if graph_type == 'variando_V1':
            fixed_values = "V2 = 6.0 V e V4 = 0.5 V fixos"
        elif graph_type == 'variando_V2':
            fixed_values = "V1 = 6.0 V e V4 = 0.5 V fixos"
        elif graph_type == 'variando_V4':
            fixed_values = "V1 = 6.0 V e V2 = 6.0 V fixos"
    
    # Título com quebra de linha: "Variando Vx" em uma linha, valores fixos na próxima
    ax.set_title(f'Experimento Franck-Hertz - Múltiplas Excitações \nTemperatura: {temp_celsius}, Variando {var_name}\ncom {fixed_values}', 
                 fontsize=16, pad=20)
    
    # Legenda
    ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
    
    # Grade
    ax.grid(True, alpha=0.3)
    
    # Limites dos eixos
    all_tensoes = np.concatenate([d['tensao'] for d in data_list])
    ax.set_xlim(0, np.max(all_tensoes) * 1.05)
    ax.set_ylim(0, None)
    
    # Melhorar layout
    plt.tight_layout()
    
    # Salvar gráfico
    output_file = os.path.join(output_dir, f'{graph_type}.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Gráfico salvo: {output_file}")

# Processar cada temperatura
for temp_folder, info in temperaturas_info.items():
    temp_dir = os.path.join(base_data_dir, temp_folder, info['coef_dir'])
    temp_celsius = info['temp_celsius']
    coef_value = info['coef_value']
    
    print(f"\nProcessando temperatura: {temp_folder}, {temp_celsius}, coeficiente: {coef_value}")
    
    if not os.path.exists(temp_dir):
        print(f"Diretório não encontrado: {temp_dir}")
        continue
    
    # Criar diretório de saída para esta temperatura
    output_temp_dir = os.path.join(base_output_dir, temp_folder)
    os.makedirs(output_temp_dir, exist_ok=True)
    
    # Listar todos os arquivos .txt
    files = [f for f in os.listdir(temp_dir) if f.endswith('.txt')]
    if not files:
        print(f"Nenhum arquivo .txt encontrado em: {temp_dir}")
        continue
    
    # Dicionários para agrupar os dados por tipo de variação
    variando_V1_data = []
    variando_V2_data = []
    variando_V4_data = []
    
    # Processar cada arquivo
    for file in files:
        file_path = os.path.join(temp_dir, file)
        v1, v2, v4 = extract_params(file)
        if v1 is None:
            continue
        
        # Ler dados do arquivo
        tensao, corrente = read_file_data(file_path)
        if tensao is None or len(tensao) == 0:
            continue
        
        # Classificar o arquivo
        classification = classify_file(v1, v2, v4, temp_folder)
        
        if classification:
            data_entry = {
                'tensao': tensao,
                'corrente': corrente,
                'v1': v1,
                'v2': v2,
                'v4': v4,
                'file': file
            }
            
            if classification == 'variando_V1':
                variando_V1_data.append(data_entry)
            elif classification == 'variando_V2':
                variando_V2_data.append(data_entry)
            elif classification == 'variando_V4':
                variando_V4_data.append(data_entry)
            
            print(f"  {file}: {classification} - V1={v1}, V2={v2}, V4={v4}")
        else:
            print(f"  {file}: não classificado - V1={v1}, V2={v2}, V4={v4}")
    
    # Gerar os 3 gráficos para esta temperatura
    print(f"\nGerando gráficos para {temp_folder}:")
    
    generate_graph(variando_V1_data, 'variando_V1', temp_folder, temp_celsius, output_temp_dir, coef_value)
    generate_graph(variando_V2_data, 'variando_V2', temp_folder, temp_celsius, output_temp_dir, coef_value)
    generate_graph(variando_V4_data, 'variando_V4', temp_folder, temp_celsius, output_temp_dir, coef_value)

print("\n" + "="*50)
print("Todos os gráficos gerados com sucesso!")
print(f"Arquivos salvos em: {base_output_dir}")
print("Estrutura criada:")
for temp in temperaturas_info.keys():
    print(f"  {temp}/")
    print(f"    ├── variando_V1.png")
    print(f"    ├── variando_V2.png")
    print(f"    └── variando_V4.png")