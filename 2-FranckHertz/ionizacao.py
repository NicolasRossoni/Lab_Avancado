import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import io

# Configurar matplotlib para melhor visualização (mesma identidade visual do uma.py)
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# Diretórios base
base_data_dir = '/home/nico/Documentos/Lab_Avancado/2-FranckHertz/Dados/Ionizacao'
base_output_dir = '/home/nico/Documentos/Lab_Avancado/2-FranckHertz/Graficos/Ionizacao'

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

# Função para extrair coeficiente da pasta (ex: 3_8 -> 3e-8)
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

# Função para extrair temperatura do nome do arquivo: {temp}.txt
def extract_temp(filename):
    """
    Extrai temperatura do nome do arquivo
    Formato: {temp}.txt (ex: 100.txt, 110.txt)
    """
    match = re.match(r'(\d+)\.txt', filename)
    if match:
        temp = int(match.group(1))
        return temp
    else:
        print(f"Nome de arquivo inválido: {filename}")
        return None

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

# Função para gerar o gráfico com todas as temperaturas
def generate_graph(all_data, output_dir):
    """
    Gera um único gráfico com todas as temperaturas da pasta Ionizacao
    """
    if not all_data:
        print("Nenhum dado encontrado para o gráfico")
        return
    
    # Criar diretório de saída
    os.makedirs(output_dir, exist_ok=True)
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    cores = ['blue', 'darkblue', 'green', 'darkgreen', 'orange', 'darkorange', 'red', 'darkred', 'purple', 'brown']
    
    for i, data in enumerate(all_data):
        tensao_valid = data['tensao']
        corrente_valid = data['corrente']
        temp_celsius = data['temp_celsius']
        filename = data['file']
        coef_value = data['coef_value']
        
        color = cores[i % len(cores)]
        
        # Aplicar conversão para nanoampères: dado * coef * 1e9
        corrente_scaled = corrente_valid * coef_value * 1e9
        
        # Ordenar por tensão para conexão correta
        sort_idx = np.argsort(tensao_valid)
        tensao_sorted = tensao_valid[sort_idx]
        corrente_sorted = corrente_scaled[sort_idx]
        
        # Legenda apenas com temperatura
        label = f'{temp_celsius} °C'
        
        # Plot dos dados conectados por linhas simples
        ax.plot(tensao_sorted, corrente_sorted, 
               color=color, linewidth=2, alpha=0.8,
               label=label,
               marker='')
        
        print(f"  {filename}: {label}, {len(tensao_sorted)} pontos")
    
    # Configurar o gráfico
    ax.set_xlabel('Tensão V3 (V)', fontsize=14)
    ax.set_ylabel('Corrente (nA)', fontsize=14)
    
    # Título específico para experimento de ionização
    ax.set_title(f'Experimento Franck-Hertz - Ionização\nV1 = 6.0V e V2 = 3.0V fixos', 
                 fontsize=16, pad=20)
    
    # Legenda
    ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
    
    # Grade
    ax.grid(True, alpha=0.3)
    
    # Limites dos eixos
    all_tensoes = np.concatenate([d['tensao'] for d in all_data])
    ax.set_xlim(10, 20)
    ax.set_ylim(0, None)
    
    # Melhorar layout
    plt.tight_layout()
    
    # Salvar gráfico
    output_file = os.path.join(output_dir, 'temperaturas.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Gráfico salvo: {output_file}")

# Processar dados de ionização
def main():
    print("Processando dados de Ionização...")
    
    # Lista de pastas de coeficiente (baseado na estrutura 3_8)
    subdir_name = '3_8'
    data_dir = os.path.join(base_data_dir, subdir_name)
    
    if not os.path.exists(data_dir):
        print(f"Diretório não encontrado: {data_dir}")
        print("Certifique-se de que os dados estão em: Dados/Ionizacao/3_8/")
        return
    
    # Listar todos os arquivos .txt
    files = [f for f in os.listdir(data_dir) if f.endswith('.txt')]
    if not files:
        print(f"Nenhum arquivo .txt encontrado em: {data_dir}")
        return
    
    print(f"Encontrados {len(files)} arquivos:")
    
    # Lista para armazenar todos os dados
    all_data = []
    
    # Extrair coeficiente da pasta
    coef_value = extract_coef_from_dir(subdir_name)
    if coef_value is None:
        print(f"Não foi possível extrair coeficiente de {subdir_name}")
        return
    
    print(f"Coeficiente extraído: {coef_value:.0e}")
    
    # Processar cada arquivo
    for file in files:
        print(f"Processando: {file}")
        
        file_path = os.path.join(data_dir, file)
        temp = extract_temp(file)
        
        if temp is None:
            continue
        
        temp_celsius = f"{temp} °C"
        
        # Ler dados do arquivo
        tensao, corrente = read_file_data(file_path)
        if tensao is None or len(tensao) == 0:
            print(f"  Nenhum dado válido encontrado, pulando...")
            continue
        
        # Adicionar à lista de dados
        data_entry = {
            'tensao': tensao,
            'corrente': corrente,
            'temp_celsius': temp_celsius,
            'file': file,
            'coef_value': coef_value
        }
        
        all_data.append(data_entry)
        print(f"  ✓ Temperatura: {temp_celsius}, {len(tensao)} pontos")
    
    if not all_data:
        print("Nenhum dado válido encontrado!")
        return
    
    # Ordenar por temperatura crescente
    all_data.sort(key=lambda x: int(x['temp_celsius'].split()[0]))
    
    # Gerar o gráfico
    print(f"\nGerando gráfico com {len(all_data)} curvas...")
    generate_graph(all_data, base_output_dir)
    
    print("\n" + "="*50)
    print("Gráfico de Ionização gerado com sucesso!")
    print(f"Arquivo salvo em: {base_output_dir}/temperaturas.png")
    print("Legenda mostra: Temperatura °C")

if __name__ == "__main__":
    main()
