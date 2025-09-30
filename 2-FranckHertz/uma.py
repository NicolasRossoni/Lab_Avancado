import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import io

# Configurar matplotlib para melhor visualização (mesma identidade visual do multiplas.py)
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# Diretórios base
base_data_dir = '/home/nico/Documentos/Lab_Avancado/2-FranckHertz/Dados/Uma'
base_output_dir = '/home/nico/Documentos/Lab_Avancado/2-FranckHertz/Graficos/Uma'

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

# Função para extrair coeficiente da pasta (ex: 1_11 -> 1e-11, 3_9 -> 3e-9)
def extract_coef_from_dir(subdir_name):
    """
    Extrai coeficiente da pasta de coeficiente
    Formato: {numero}_{expoente} -> numero * 10^(-expoente)
    """
    match = re.match(r'(\d+)_(\d+)', subdir_name)
    if match:
        numero, expoente = map(int, match.groups())
        coef_value = numero * (10 ** -expoente)
        return coef_value
    else:
        print(f"Formato de pasta inválido: {subdir_name}")
        return None

# Função para extrair parâmetros do nome do arquivo: {Temp}_{V1}_{V4}.txt
def extract_params(filename):
    """
    Extrai temperatura, V1 e V4 do nome do arquivo
    Formato: {Temp}_{V1}_{V4}.txt (ex: 110_6.0_1.5.txt)
    """
    match = re.match(r'(\d+)_(\d+(?:\.\d+)?)_(\d+(?:\.\d+)?)\.txt', filename)
    if match:
        temp_str, v1, v4 = match.groups()
        temp_kelvin = float(temp_str)
        v1 = float(v1)
        v4 = float(v4)
        return temp_str, v1, v4, temp_kelvin
    else:
        print(f"Nome de arquivo inválido: {filename}")
        return None, None, None, None

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
                       sep=r'\s+', header=None,
                       names=['tensao', 'corrente'])
        
        tensao = df['tensao'].apply(convert_to_float).values
        corrente = df['corrente'].apply(convert_to_float).values
        
        valid_idx = (~np.isnan(tensao) & ~np.isnan(corrente) & 
                    np.isfinite(tensao) & np.isfinite(corrente))
        
        return tensao[valid_idx], corrente[valid_idx]
        
    except Exception as e:
        print(f"Erro ao ler {file_path}: {e}")
        return None, None

# Função para gerar o gráfico para uma pasta específica
def generate_graph_for_coef(data_list, coef_dir, output_dir):
    """
    Gera um gráfico para uma pasta específica de coeficiente
    """
    if not data_list:
        print(f"Nenhum dado encontrado para {coef_dir}")
        return
    
    # Criar diretório de saída
    os.makedirs(output_dir, exist_ok=True)
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    cores = ['blue', 'darkblue', 'green', 'darkgreen', 'orange', 'darkorange', 'red', 'darkred', 'purple', 'brown']
    
    for i, data in enumerate(data_list):
        tensao_valid = data['tensao']
        corrente_valid = data['corrente']
        temp_str = data['temp_str']
        temp_kelvin = data['temp_kelvin']
        temp_celsius = data['temp_celsius']
        v1 = data['v1']
        v4 = data['v4']
        filename = data['file']
        coef_value = data['coef_value']
        
        color = cores[i % len(cores)]
        
        # Aplicar conversão para nanoampères: dado * coef * 1e9
        corrente_scaled = corrente_valid * coef_value * 1e9 / 100
        
        # Ordenar por tensão para conexão correta
        sort_idx = np.argsort(tensao_valid)
        tensao_sorted = tensao_valid[sort_idx]
        corrente_sorted = corrente_scaled[sort_idx]
        
        # Legenda com temperatura, V1 e V4
        label = f'{temp_celsius} (V1={v1}V, V4={v4}V)'
        
        # Plot dos dados conectados por linhas simples
        ax.plot(tensao_sorted, corrente_sorted, 
               color=color, linewidth=2, alpha=0.8,
               label=label,
               marker='')
        
        print(f"  {filename}: {label}, {len(tensao_sorted)} pontos")
    
    # Configurar o gráfico
    ax.set_xlabel('Tensão V3 (V)', fontsize=14)
    ax.set_ylabel('Corrente (nA)', fontsize=14)
    
    # Título específico para experimento de uma excitação
    ax.set_title(f'Experimento Franck-Hertz - Uma Excitação para Várias Temperaturas', 
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
    
    # Salvar gráfico com nome específico da pasta
    output_file = os.path.join(output_dir, f'temperaturas_{coef_dir}.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Gráfico salvo: {output_file}")

# Processar dados de uma excitação
def main():
    print("Processando dados de Uma Excitação (1_11 e 3_9 separadamente)...")
    
    # Lista de pastas de coeficiente a processar
    coef_dirs = ['1_11', '3_9']
    
    # Lista para armazenar todos os dados de cada pasta
    all_data_by_coef = {}
    
    # Processar cada pasta de coeficiente
    for coef_dir in coef_dirs:
        print(f"\n--- Processando pasta: {coef_dir} ---")
        
        data_dir = os.path.join(base_data_dir, coef_dir)
        if not os.path.exists(data_dir):
            print(f"Diretório não encontrado: {data_dir}")
            continue
        
        # Listar todos os arquivos .txt
        files = [f for f in os.listdir(data_dir) if f.endswith('.txt')]
        if not files:
            print(f"Nenhum arquivo .txt encontrado em: {data_dir}")
            continue
        
        print(f"Encontrados {len(files)} arquivos em {coef_dir}:")
        
        # Extrair coeficiente da pasta
        coef_value = extract_coef_from_dir(coef_dir)
        if coef_value is None:
            print(f"Não foi possível extrair coeficiente de {coef_dir}")
            continue
        
        print(f"Coeficiente extraído: {coef_value:.0e}")
        
        # Lista específica para esta pasta
        data_list = []
        
        # Processar cada arquivo
        for file in files:
            print(f"Processando: {file}")
            
            file_path = os.path.join(data_dir, file)
            temp_str, v1, v4, temp_kelvin = extract_params(file)
            
            if temp_str is None:
                continue
            
            # Converter temperatura para Celsius (subtrair 273 se necessário)
            # Como os nomes dos arquivos já indicam Celsius, manter como está
            temp_celsius = f"{int(temp_kelvin)} °C"
            
            # Ler dados do arquivo
            tensao, corrente = read_file_data(file_path)
            if tensao is None or len(tensao) == 0:
                print(f"  Nenhum dado válido encontrado, pulando...")
                continue
            
            # Adicionar à lista de dados
            data_entry = {
                'tensao': tensao,
                'corrente': corrente,
                'temp_str': temp_str,
                'temp_kelvin': temp_kelvin,
                'temp_celsius': temp_celsius,
                'v1': v1,
                'v4': v4,
                'coef_value': coef_value,
                'file': file
            }
            
            data_list.append(data_entry)
            print(f"  ✓ Temperatura: {temp_celsius}, V1={v1}V, V4={v4}V, {len(tensao)} pontos")
        
        if data_list:
            all_data_by_coef[coef_dir] = data_list
            # Ordenar por temperatura crescente para esta pasta
            data_list.sort(key=lambda x: x['temp_kelvin'])
        else:
            print(f"Nenhum dado válido encontrado em {coef_dir}")
    
    # Gerar gráficos separados para cada pasta
    for coef_dir, data_list in all_data_by_coef.items():
        print(f"\nGerando gráfico para {coef_dir} com {len(data_list)} curvas...")
        generate_graph_for_coef(data_list, coef_dir, base_output_dir)
    
    if not all_data_by_coef:
        print("Nenhum dado válido encontrado em nenhuma pasta!")
        return
    
    print("\n" + "="*50)
    print("Gráficos de Uma Excitação gerados com sucesso!")
    print(f"Arquivos salvos em: {base_output_dir}/")
    for coef_dir in all_data_by_coef.keys():
        print(f"  ├── temperaturas_{coef_dir}.png")

if __name__ == "__main__":
    main()
