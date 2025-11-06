"""
analise_interativa.py

Aplicativo Pygame para análise interativa das imagens de difração.
Permite navegar pelas fotos 15-50V e definir valores CM, R1 e R2 clicando na imagem.
"""

import pygame
import sys
import os
from pathlib import Path
import numpy as np
from PIL import Image
import csv
import matplotlib.pyplot as plt
import matplotlib.backends.backend_agg as agg
import io

# =========================
# CONFIGURAÇÕES
# =========================
PIXEL_POR_CM = 80
CM_POR_PIXEL = 1/PIXEL_POR_CM
CENTER_X = 621  # Centro padrão (pode ser redefinido)
CENTER_Y = 463

# Configurações da janela
WINDOW_WIDTH = 1400
WINDOW_HEIGHT = 900
IMAGE_WIDTH = 900
IMAGE_HEIGHT = 450
PANEL_WIDTH = 400
TABLE_HEIGHT = 400
VISIBLE_ROWS = 10

# Cores
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (128, 128, 128)
LIGHT_GRAY = (200, 200, 200)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
CYAN = (0, 255, 255)

# =========================
# CLASSE PRINCIPAL
# =========================
class AnaliseDifracao:
    def __init__(self):
        pygame.init()
        
        # Configurar janela
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Análise Interativa - Difração de Elétrons")
        
        # Fontes
        self.font_title = pygame.font.Font(None, 32)
        self.font_text = pygame.font.Font(None, 24)
        self.font_small = pygame.font.Font(None, 20)
        
        # Configurar paths
        self.project_dir = Path(__file__).parent
        self.data_dir = self.project_dir / "Data" / "raw_images"
        self.csv_path = self.project_dir / "dados_analise.csv"
        
        # Estado da aplicação
        self.current_voltage = 15  # Começar com 15V
        self.voltages = list(range(15, 51))  # 15 a 50V
        self.current_image = None
        self.current_image_surface = None
        self.image_rect = pygame.Rect(50, 50, IMAGE_WIDTH, IMAGE_HEIGHT)
        
        # Modo de seleção
        self.selection_mode = "CM1"  # "CM1", "CM2", "R1", "R2"
        self.cm_mode = 1  # 1 para CM1, 2 para CM2
        self.selection_state = {}
        
        # Inicializar dados para todas as voltagens
        for voltage in self.voltages:
            key = f"{voltage}V"
            self.selection_state[key] = {
                "CM1": None,
                "CM2": None,
                "r1": None, "R1": None,
                "r2": None, "R2": None
            }
        
        # Estado de seleção para R1/R2
        self.r_selection_step = 0  # 0: selecionar r (minúsculo), 1: selecionar R (maiúsculo)
        
        # Rastrear último dado editado para destaque na tabela
        self.last_edited = {"voltage": None, "field": None}
        
        # Controle de scroll da tabela
        self.table_scroll_offset = 0
        self.max_scroll_offset = max(0, len(self.voltages) - VISIBLE_ROWS)
        
        # Navegação WASD por células da tabela
        self.selected_row = 0  # Índice na lista de voltagens (0-35)
        self.selected_col = 1  # Coluna: 0=Volts, 1=CM1, 2=CM2, 3=r1, 4=R1, 5=r2, 6=R2
        self.table_columns = ["Volts", "CM1", "CM2", "r1", "R1", "r2", "R2"]
        
        # Controle da linha amarela de seleção (independente do eixo)
        self.selection_x_position = 0.0  # Posição em cm da linha amarela (inicia no centro)
        # O eixo X sempre usa o centro fixo definido por CM1/CM2
        
        # Carregar dados existentes se disponíveis
        self.load_csv_data()
        
        # Carregar primeira imagem
        self.load_current_image()
        
        print("🎮 Aplicativo iniciado!")
        print("📋 Controles:")
        print("  ← → : Mover linha amarela (seleção)")
        print("  Enter: Medir posição da linha amarela")
        print("  PgUp/PgDn : Navegar voltagens")
        print("  ↑ ↓ : Scroll da tabela")
        print("  WASD: Navegar células (W↑ A← D→, use ↓ para baixo)")
        print("  Clique: Definir valor da célula selecionada")
        print("  0   : Modo definir centro (CM1/CM2)")
        print("  Caps: Alternar CM1 ↔ CM2")
        print("  1   : Modo definir R1 (r1 + R1)")
        print("  2   : Modo definir R2 (r2 + R2)")
        print("  S   : Salvar dados em CSV")
        print("  ESC : Sair")
    
    def method3_green_channel(self, img_array: np.ndarray) -> np.ndarray:
        """Método 3: Apenas canal verde (ideal para padrões verdes)."""
        return img_array[:,:,1] / 255.0
    
    def generate_green_channel_plot(self):
        """Gera gráfico do canal verde com imagem de fundo."""
        image_file = f"{self.current_voltage}.jpeg"
        image_path = self.data_dir / image_file
        
        if not image_path.exists():
            print(f"❌ Imagem não encontrada: {image_path}")
            return None
        
        # Carregar imagem original
        img = Image.open(image_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        original_image = np.array(img)
        
        # Usar canal verde
        intensity_array = self.method3_green_channel(original_image)
        
        # Coordenadas
        height, width = original_image.shape[:2]
        
        # Usar sempre o centro fixo da imagem (CENTER_X)
        # CM1 e CM2 são apenas parâmetros medidos, não afetam o eixo
        center_x = CENTER_X  # Sempre fixo!
        
        x_pixels = np.arange(width) - center_x
        x_cm = x_pixels * CM_POR_PIXEL
        y_pixels = np.arange(height) - CENTER_Y
        y_cm = y_pixels * CM_POR_PIXEL
        extent = [x_cm[0], x_cm[-1], y_cm[-1], y_cm[0]]
        
        # Perfil horizontal
        horizontal_line = intensity_array[CENTER_Y, :]
        
        # Criar figura matplotlib (menor para o novo layout)
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        # Mostrar imagem original como fundo
        ax.imshow(original_image, extent=extent, aspect='auto', alpha=0.7, 
                  origin='upper', interpolation='bilinear')
        
        # Configurações do eixo Y (só parte superior)
        y_range = 6.0
        ax.set_ylim(0, y_range)
        
        # Marcar linha analisada horizontal
        ax.axhline(y=0, color='red', linestyle='-', linewidth=2, alpha=0.8)
        
        # Marcar linha amarela de seleção (sempre relativa ao eixo fixo)
        ax.axvline(x=self.selection_x_position, color='yellow', linestyle='-', linewidth=3, alpha=0.9, 
                   label=f'Seleção: {self.selection_x_position:.3f}cm')
        
        # Marcar centro branco (eixo X fixo)
        ax.axvline(x=0, color='white', linewidth=3, alpha=0.9, zorder=11, label='Centro (0,0)')
        ax.plot(0, 0, 'o', color='yellow', markersize=10, markeredgewidth=3, 
                markerfacecolor='yellow', markeredgecolor='red', 
                label=f'Centro ({center_x:.0f}, {CENTER_Y})px', zorder=12)
        
        # Eixo secundário para intensidade
        ax2 = ax.twinx()
        
        # Curva verde: Intensidade (usar o array x_cm original, não o valor único)
        x_cm_array = np.arange(width) - center_x
        x_cm_array = x_cm_array * CM_POR_PIXEL
        ax2.plot(x_cm_array, horizontal_line, color='lime', linewidth=4, alpha=0.9,
                 label='Intensidade (Canal Verde)')
        
        # Configurar eixo Y
        ax2.set_ylabel('Intensidade', fontsize=12, fontweight='bold')
        ax2.tick_params(axis='y', labelcolor='black')
        ax2.set_ylim(0, horizontal_line.max() * 1.1)
        
        # Configurações
        ax.set_xlabel('Distância do Centro (cm)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Posição Y (cm)', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.2)
        
        # Título
        title = f'Canal Verde - {self.current_voltage}V - Centro: CM{self.cm_mode}'
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20,
                    bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.9))
        
        # Legenda
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        all_lines = lines1 + lines2
        all_labels = labels1 + labels2
        ax.legend(all_lines, all_labels, loc='upper left', fontsize=10, 
                  bbox_to_anchor=(0.02, 0.98), framealpha=0.9)
        
        plt.tight_layout()
        
        # Converter para pygame surface - método compatível
        canvas = agg.FigureCanvasAgg(fig)
        canvas.draw()
        
        # Obter dados do buffer
        buf = canvas.buffer_rgba()
        w, h = canvas.get_width_height()
        
        # Converter RGBA para RGB
        arr = np.frombuffer(buf, dtype=np.uint8).reshape((h, w, 4))
        arr_rgb = arr[:, :, :3]  # Remover canal alpha
        
        # Criar surface pygame
        surf = pygame.surfarray.make_surface(arr_rgb.swapaxes(0, 1))
        
        plt.close(fig)  # Fechar figura para liberar memória
        
        return surf
    
    def load_current_image(self):
        """Gera o gráfico do canal verde para a voltagem atual."""
        self.current_image_surface = self.generate_green_channel_plot()
        if self.current_image_surface:
            # Redimensionar para caber na área da imagem
            self.current_image_surface = pygame.transform.scale(
                self.current_image_surface, (IMAGE_WIDTH, IMAGE_HEIGHT))
            print(f"✅ Gráfico gerado: {self.current_voltage}V")
        else:
            print(f"❌ Erro ao gerar gráfico: {self.current_voltage}V")
    
    def pixel_to_cm(self, pixel_x):
        """Converte coordenada de pixel para centímetros a partir do centro."""
        # Ajustar para o redimensionamento da imagem
        scale_x = 1280 / IMAGE_WIDTH  # Imagem original tem 1280px de largura
        real_pixel_x = pixel_x * scale_x
        
        # Usar centro atual (CM1 ou CM2)
        voltage_key = f"{self.current_voltage}V"
        center_key = f"CM{self.cm_mode}"
        if self.selection_state[voltage_key][center_key] is not None:
            center_x = self.selection_state[voltage_key][center_key]
        else:
            center_x = CENTER_X
            
        cm_x = (real_pixel_x - center_x) * CM_POR_PIXEL
        return cm_x
    
    def handle_click(self, pos):
        """Processa clique do mouse na imagem."""
        if not self.image_rect.collidepoint(pos):
            return
        
        # Converter posição do clique para coordenadas da imagem
        image_x = pos[0] - self.image_rect.x
        image_y = pos[1] - self.image_rect.y
        
        # Converter para centímetros
        cm_x = self.pixel_to_cm(image_x)
        
        # Obter informações da célula selecionada (modo WASD)
        selected_voltage, selected_column = self.get_selected_cell_info()
        voltage_key = f"{selected_voltage}V"
        
        # Modo WASD: usar célula selecionada
        if selected_column in ["CM1", "CM2"]:
            # Definir novo centro
            scale_x = 1280 / IMAGE_WIDTH
            new_center = image_x * scale_x
            self.selection_state[voltage_key][selected_column] = new_center
            self.last_edited = {"voltage": selected_voltage, "field": selected_column}
            print(f"🎯 {selected_column} definido para {selected_voltage}V: {new_center:.1f}px")
            # Se é a voltagem atual, regenerar gráfico
            if selected_voltage == self.current_voltage:
                self.load_current_image()
        
        elif selected_column in ["r1", "R1", "r2", "R2"]:
            # Definir coordenada em cm
            self.selection_state[voltage_key][selected_column] = cm_x
            self.last_edited = {"voltage": selected_voltage, "field": selected_column}
            print(f"📍 {selected_column} definido para {selected_voltage}V: {cm_x:.3f}cm")
        
        # Modo legado (manter compatibilidade)
        voltage_key_current = f"{self.current_voltage}V"
        
        if self.selection_mode in ["CM1", "CM2"]:
            # Definir novo centro (CM1 ou CM2)
            scale_x = 1280 / IMAGE_WIDTH
            new_center = image_x * scale_x
            center_key = f"CM{self.cm_mode}"
            self.selection_state[voltage_key_current][center_key] = new_center
            self.last_edited = {"voltage": self.current_voltage, "field": center_key}
            print(f"🎯 [Modo Legado] {center_key} definido para {self.current_voltage}V: {new_center:.1f}px")
            # Regenerar gráfico com novo centro
            self.load_current_image()
            
        elif self.selection_mode == "R1":
            if self.r_selection_step == 0:
                # Selecionar r1 (minúsculo)
                self.selection_state[voltage_key_current]["r1"] = cm_x
                self.last_edited = {"voltage": self.current_voltage, "field": "r1"}
                self.r_selection_step = 1
                print(f"📍 [Modo Legado] r1 definido para {self.current_voltage}V: {cm_x:.3f}cm")
            else:
                # Selecionar R1 (maiúsculo)
                self.selection_state[voltage_key_current]["R1"] = cm_x
                self.last_edited = {"voltage": self.current_voltage, "field": "R1"}
                self.r_selection_step = 0
                print(f"📍 [Modo Legado] R1 definido para {self.current_voltage}V: {cm_x:.3f}cm")
                
        elif self.selection_mode == "R2":
            if self.r_selection_step == 0:
                # Selecionar r2 (minúsculo)
                self.selection_state[voltage_key_current]["r2"] = cm_x
                self.last_edited = {"voltage": self.current_voltage, "field": "r2"}
                self.r_selection_step = 1
                print(f"📍 [Modo Legado] r2 definido para {self.current_voltage}V: {cm_x:.3f}cm")
            else:
                # Selecionar R2 (maiúsculo)
                self.selection_state[voltage_key_current]["R2"] = cm_x
                self.last_edited = {"voltage": self.current_voltage, "field": "R2"}
                self.r_selection_step = 0
                print(f"📍 [Modo Legado] R2 definido para {self.current_voltage}V: {cm_x:.3f}cm")
    
    def change_voltage(self, direction):
        """Muda a voltagem atual."""
        current_index = self.voltages.index(self.current_voltage)
        
        if direction == "next" and current_index < len(self.voltages) - 1:
            self.current_voltage = self.voltages[current_index + 1]
            self.selected_row = current_index + 1  # Sincronizar WASD
            self.load_current_image()
            self.r_selection_step = 0  # Reset selection step
            self.ensure_current_voltage_visible()
            self.ensure_selected_cell_visible()
        elif direction == "prev" and current_index > 0:
            self.current_voltage = self.voltages[current_index - 1]
            self.selected_row = current_index - 1  # Sincronizar WASD
            self.load_current_image()
            self.r_selection_step = 0  # Reset selection step
            self.ensure_current_voltage_visible()
            self.ensure_selected_cell_visible()
    
    def toggle_cm_mode(self):
        """Alterna entre CM1 e CM2."""
        self.cm_mode = 2 if self.cm_mode == 1 else 1
        self.selection_mode = f"CM{self.cm_mode}"
        print(f"🔄 Alternado para: CM{self.cm_mode}")
        # Regenerar gráfico com novo centro
        self.load_current_image()
    
    def change_mode(self, mode):
        """Muda o modo de seleção."""
        if mode == "CM":
            self.selection_mode = f"CM{self.cm_mode}"
        else:
            self.selection_mode = mode
        self.r_selection_step = 0  # Reset selection step
        print(f"🔄 Modo alterado para: {self.selection_mode}")
    
    def scroll_table(self, direction):
        """Controla o scroll da tabela."""
        if direction == "up" and self.table_scroll_offset > 0:
            self.table_scroll_offset -= 1
        elif direction == "down" and self.table_scroll_offset < self.max_scroll_offset:
            self.table_scroll_offset += 1
    
    def ensure_current_voltage_visible(self):
        """Garante que a voltagem atual esteja visível na tabela."""
        current_index = self.voltages.index(self.current_voltage)
        
        # Se a linha atual está acima da área visível
        if current_index < self.table_scroll_offset:
            self.table_scroll_offset = current_index
        
        # Se a linha atual está abaixo da área visível
        elif current_index >= self.table_scroll_offset + VISIBLE_ROWS:
            self.table_scroll_offset = current_index - VISIBLE_ROWS + 1
            self.table_scroll_offset = max(0, self.table_scroll_offset)
    
    def ensure_selected_cell_visible(self):
        """Garante que a célula selecionada esteja visível na tabela."""
        # Se a linha selecionada está acima da área visível
        if self.selected_row < self.table_scroll_offset:
            self.table_scroll_offset = self.selected_row
        
        # Se a linha selecionada está abaixo da área visível
        elif self.selected_row >= self.table_scroll_offset + VISIBLE_ROWS:
            self.table_scroll_offset = self.selected_row - VISIBLE_ROWS + 1
            self.table_scroll_offset = max(0, self.table_scroll_offset)
    
    def navigate_table_cell(self, direction):
        """Navega pelas células da tabela com WASD."""
        if direction == "up" and self.selected_row > 0:
            self.selected_row -= 1
            self.current_voltage = self.voltages[self.selected_row]  # Atualizar voltagem
            self.ensure_selected_cell_visible()
            self.load_current_image()  # Regenerar gráfico
            print(f"📍 Célula: {self.voltages[self.selected_row]}V - {self.table_columns[self.selected_col]}")
        
        elif direction == "down" and self.selected_row < len(self.voltages) - 1:
            self.selected_row += 1
            self.current_voltage = self.voltages[self.selected_row]  # Atualizar voltagem
            self.ensure_selected_cell_visible()
            self.load_current_image()  # Regenerar gráfico
            print(f"📍 Célula: {self.voltages[self.selected_row]}V - {self.table_columns[self.selected_col]}")
        
        elif direction == "left" and self.selected_col > 1:  # Não selecionar coluna "Volts"
            self.selected_col -= 1
            print(f"📍 Célula: {self.voltages[self.selected_row]}V - {self.table_columns[self.selected_col]}")
        
        elif direction == "right" and self.selected_col < len(self.table_columns) - 1:
            self.selected_col += 1
            print(f"📍 Célula: {self.voltages[self.selected_row]}V - {self.table_columns[self.selected_col]}")
    
    def get_selected_cell_info(self):
        """Retorna informações da célula atualmente selecionada."""
        voltage = self.voltages[self.selected_row]
        column = self.table_columns[self.selected_col]
        return voltage, column
    
    def move_selection_line(self, direction):
        """Move a linha amarela de seleção para esquerda/direita."""
        step_cm = 0.1  # Movimento em cm
        
        if direction == "left":
            self.selection_x_position -= step_cm
            print(f"📍 Linha amarela: {self.selection_x_position:.3f}cm")
            self.load_current_image()  # Regenerar gráfico com nova posição
        
        elif direction == "right":
            self.selection_x_position += step_cm
            print(f"📍 Linha amarela: {self.selection_x_position:.3f}cm")
            self.load_current_image()  # Regenerar gráfico com nova posição
    
    # Função removida - centro sempre fixo em CENTER_X
    
    def make_measurement_at_selection(self):
        """Faz uma medida na posição da linha amarela e salva no campo selecionado."""
        # Obter informações da célula selecionada
        selected_voltage, selected_column = self.get_selected_cell_info()
        voltage_key = f"{selected_voltage}V"
        
        # Usar posição da linha amarela
        selection_cm = self.selection_x_position
        
        # Salvar no campo apropriado
        if selected_column in ["CM1", "CM2"]:
            # Para centros, converter de cm para pixels usando centro fixo
            selection_px = CENTER_X + (selection_cm / CM_POR_PIXEL)
            self.selection_state[voltage_key][selected_column] = selection_px
            self.last_edited = {"voltage": selected_voltage, "field": selected_column}
            print(f"📏 {selected_column} medido para {selected_voltage}V: {selection_px:.1f}px ({selection_cm:.3f}cm)")
            # Não precisa regenerar gráfico - centro sempre fixo!
        
        elif selected_column in ["r1", "R1", "r2", "R2"]:
            # Para coordenadas, salvar em cm
            self.selection_state[voltage_key][selected_column] = selection_cm
            self.last_edited = {"voltage": selected_voltage, "field": selected_column}
            print(f"📏 {selected_column} medido para {selected_voltage}V: {selection_cm:.3f}cm")
        
        else:
            print(f"⚠️  Não é possível medir na coluna '{selected_column}'")
    
    def save_csv_data(self):
        """Salva os dados coletados em CSV."""
        try:
            with open(self.csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['Volts', 'CM1', 'CM2', 'r1', 'R1', 'r2', 'R2']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                writer.writeheader()
                
                for voltage in self.voltages:
                    voltage_key = f"{voltage}V"
                    data = self.selection_state[voltage_key]
                    
                    row = {
                        'Volts': voltage,
                        'CM1': data.get('CM1', ''),
                        'CM2': data.get('CM2', ''),
                        'r1': data.get('r1', ''),
                        'R1': data.get('R1', ''),
                        'r2': data.get('r2', ''),
                        'R2': data.get('R2', '')
                    }
                    writer.writerow(row)
            
            print(f"💾 Dados salvos em: {self.csv_path}")
            
        except Exception as e:
            print(f"❌ Erro ao salvar CSV: {e}")
    
    def load_csv_data(self):
        """Carrega dados existentes do CSV."""
        if not self.csv_path.exists():
            return
        
        try:
            with open(self.csv_path, 'r', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                
                for row in reader:
                    voltage = int(row['Volts'])
                    if voltage in self.voltages:
                        voltage_key = f"{voltage}V"
                        
                        # Carregar apenas valores não vazios
                        if row.get('CM1') and row['CM1'].strip():
                            self.selection_state[voltage_key]['CM1'] = float(row['CM1'])
                        if row.get('CM2') and row['CM2'].strip():
                            self.selection_state[voltage_key]['CM2'] = float(row['CM2'])
                        if row.get('CM') and row['CM'].strip():  # Compatibilidade com CSV antigo
                            self.selection_state[voltage_key]['CM1'] = float(row['CM'])
                        if row.get('r1') and row['r1'].strip():
                            self.selection_state[voltage_key]['r1'] = float(row['r1'])
                        if row.get('R1') and row['R1'].strip():
                            self.selection_state[voltage_key]['R1'] = float(row['R1'])
                        if row.get('r2') and row['r2'].strip():
                            self.selection_state[voltage_key]['r2'] = float(row['r2'])
                        if row.get('R2') and row['R2'].strip():
                            self.selection_state[voltage_key]['R2'] = float(row['R2'])
            
            print(f"📂 Dados carregados do CSV: {self.csv_path}")
            
        except Exception as e:
            print(f"⚠️  Erro ao carregar CSV: {e}")
    
    def draw_csv_table(self):
        """Desenha a tabela com dados do CSV com scroll."""
        table_start_y = IMAGE_HEIGHT + 80
        
        # Título da tabela
        title_text = self.font_title.render("📊 Dados Coletados - CSV", True, BLACK)
        self.screen.blit(title_text, (50, table_start_y - 35))
        
        # Legenda e informações de scroll
        legend_text = self.font_small.render("🟢 Linha atual | 🔴 Último editado | 🟡 WASD selecionado | ↑↓ Scroll", True, GRAY)
        self.screen.blit(legend_text, (WINDOW_WIDTH - 500, table_start_y - 20))
        
        # Informação de scroll
        scroll_info = f"Linhas {self.table_scroll_offset + 1}-{min(self.table_scroll_offset + VISIBLE_ROWS, len(self.voltages))} de {len(self.voltages)}"
        scroll_text = self.font_small.render(scroll_info, True, GRAY)
        self.screen.blit(scroll_text, (50, table_start_y - 20))
        
        table_rect = pygame.Rect(50, table_start_y, WINDOW_WIDTH - 100, TABLE_HEIGHT)
        
        # Fundo da tabela
        pygame.draw.rect(self.screen, WHITE, table_rect)
        pygame.draw.rect(self.screen, BLACK, table_rect, 2)
        
        # Cabeçalhos
        headers = ["Volts", "CM1", "CM2", "r1", "R1", "r2", "R2"]
        col_width = (table_rect.width - 60) // len(headers)  # Espaço para scrollbar
        header_height = 35
        
        # Desenhar cabeçalhos
        for i, header in enumerate(headers):
            x = table_rect.x + 10 + i * col_width
            y = table_rect.y + 10
            
            # Fundo do cabeçalho
            header_rect = pygame.Rect(x, y, col_width, header_height)
            pygame.draw.rect(self.screen, LIGHT_GRAY, header_rect)
            pygame.draw.rect(self.screen, BLACK, header_rect, 1)
            
            # Texto do cabeçalho
            header_text = self.font_small.render(header, True, BLACK)
            text_rect = header_text.get_rect(center=header_rect.center)
            self.screen.blit(header_text, text_rect)
        
        # Dados das linhas (apenas as visíveis)
        row_height = 30
        start_data_y = table_rect.y + 10 + header_height
        
        # Calcular quais linhas mostrar
        start_row = self.table_scroll_offset
        end_row = min(start_row + VISIBLE_ROWS, len(self.voltages))
        
        for visible_idx, actual_idx in enumerate(range(start_row, end_row)):
            voltage = self.voltages[actual_idx]
            voltage_key = f"{voltage}V"
            data = self.selection_state[voltage_key]
            
            y = start_data_y + visible_idx * row_height
            
            # Dados da linha
            row_data = [
                str(voltage),
                f"{data.get('CM1', ''):.1f}" if isinstance(data.get('CM1'), (int, float)) else "",
                f"{data.get('CM2', ''):.1f}" if isinstance(data.get('CM2'), (int, float)) else "",
                f"{data.get('r1', ''):.3f}" if isinstance(data.get('r1'), (int, float)) else "",
                f"{data.get('R1', ''):.3f}" if isinstance(data.get('R1'), (int, float)) else "",
                f"{data.get('r2', ''):.3f}" if isinstance(data.get('r2'), (int, float)) else "",
                f"{data.get('R2', ''):.3f}" if isinstance(data.get('R2'), (int, float)) else ""
            ]
            
            for col_idx, value in enumerate(row_data):
                x = table_rect.x + 10 + col_idx * col_width
                cell_rect = pygame.Rect(x, y, col_width, row_height)
                
                # Cor de fundo da célula
                bg_color = WHITE
                
                # Destacar linha atual (voltagem selecionada)
                if voltage == self.current_voltage:
                    bg_color = (200, 255, 200)  # Verde claro
                
                # Destacar célula selecionada com WASD
                if (actual_idx == self.selected_row and col_idx == self.selected_col):
                    bg_color = (255, 255, 150)  # Amarelo claro
                
                # Destacar célula editada recentemente
                if (self.last_edited["voltage"] == voltage and 
                    col_idx > 0 and headers[col_idx] == self.last_edited["field"]):
                    bg_color = (255, 200, 200)  # Vermelho claro
                
                pygame.draw.rect(self.screen, bg_color, cell_rect)
                pygame.draw.rect(self.screen, BLACK, cell_rect, 1)
                
                # Borda especial para célula selecionada WASD
                if (actual_idx == self.selected_row and col_idx == self.selected_col):
                    pygame.draw.rect(self.screen, BLUE, cell_rect, 3)
                
                # Quadrado vermelho para último editado
                elif (self.last_edited["voltage"] == voltage and 
                    col_idx > 0 and headers[col_idx] == self.last_edited["field"]):
                    pygame.draw.rect(self.screen, RED, cell_rect, 3)
                
                # Texto da célula
                if value:
                    text_surface = self.font_small.render(value, True, BLACK)
                    text_rect = text_surface.get_rect(center=cell_rect.center)
                    self.screen.blit(text_surface, text_rect)
        
        # Desenhar scrollbar
        if len(self.voltages) > VISIBLE_ROWS:
            scrollbar_x = table_rect.right - 30
            scrollbar_y = table_rect.y + 10 + header_height
            scrollbar_height = TABLE_HEIGHT - header_height - 20
            
            # Fundo da scrollbar
            scrollbar_bg = pygame.Rect(scrollbar_x, scrollbar_y, 20, scrollbar_height)
            pygame.draw.rect(self.screen, GRAY, scrollbar_bg)
            pygame.draw.rect(self.screen, BLACK, scrollbar_bg, 1)
            
            # Handle da scrollbar
            handle_height = max(20, int(scrollbar_height * VISIBLE_ROWS / len(self.voltages)))
            handle_y = scrollbar_y + int(scrollbar_height * self.table_scroll_offset / len(self.voltages))
            handle_rect = pygame.Rect(scrollbar_x + 2, handle_y, 16, handle_height)
            pygame.draw.rect(self.screen, BLACK, handle_rect)
            pygame.draw.rect(self.screen, WHITE, handle_rect, 1)
    
    def draw_interface(self):
        """Desenha a interface completa."""
        self.screen.fill(WHITE)
        
        # Desenhar imagem (área menor no topo)
        if self.current_image_surface:
            self.screen.blit(self.current_image_surface, self.image_rect)
            
            # Desenhar borda da imagem
            pygame.draw.rect(self.screen, BLACK, self.image_rect, 2)
        
        # Painel lateral (menor)
        panel_rect = pygame.Rect(IMAGE_WIDTH + 100, 50, PANEL_WIDTH, IMAGE_HEIGHT)
        pygame.draw.rect(self.screen, LIGHT_GRAY, panel_rect)
        pygame.draw.rect(self.screen, BLACK, panel_rect, 2)
        
        # Título do painel
        title_text = self.font_title.render(f"Voltagem: {self.current_voltage}V", True, BLACK)
        self.screen.blit(title_text, (panel_rect.x + 10, panel_rect.y + 10))
        
        # Modo atual
        mode_color = RED if self.selection_mode in ["CM1", "CM2"] else GREEN if self.selection_mode == "R1" else BLUE
        mode_text = self.font_text.render(f"Modo: {self.selection_mode}", True, mode_color)
        self.screen.blit(mode_text, (panel_rect.x + 10, panel_rect.y + 50))
        
        # Estado da seleção para R1/R2
        if self.selection_mode in ["R1", "R2"]:
            step_text = "Selecionar r (minúsculo)" if self.r_selection_step == 0 else "Selecionar R (maiúsculo)"
            step_surface = self.font_small.render(step_text, True, BLACK)
            self.screen.blit(step_surface, (panel_rect.x + 10, panel_rect.y + 80))
        
        # Informação da célula selecionada WASD
        selected_voltage, selected_column = self.get_selected_cell_info()
        wasd_text = f"WASD: {selected_voltage}V - {selected_column}"
        wasd_surface = self.font_small.render(wasd_text, True, BLUE)
        self.screen.blit(wasd_surface, (panel_rect.x + 10, panel_rect.y + 100))
        
        # Informação da posição da linha amarela de seleção
        selection_text = f"Linha amarela: {self.selection_x_position:.3f}cm"
        selection_surface = self.font_small.render(selection_text, True, (255, 165, 0))  # Laranja
        self.screen.blit(selection_surface, (panel_rect.x + 10, panel_rect.y + 120))
        
        # Dados atuais
        voltage_key = f"{self.current_voltage}V"
        data = self.selection_state[voltage_key]
        
        y_offset = 140
        
        # Centros
        cm1_value = data.get('CM1', 'Não definido')
        if isinstance(cm1_value, (int, float)):
            cm1_text = f"CM1: {cm1_value:.1f}px"
        else:
            cm1_text = f"CM1: {cm1_value}"
        cm1_color = RED if self.cm_mode == 1 else GRAY
        cm1_surface = self.font_text.render(cm1_text, True, cm1_color)
        self.screen.blit(cm1_surface, (panel_rect.x + 10, panel_rect.y + y_offset))
        y_offset += 30
        
        cm2_value = data.get('CM2', 'Não definido')
        if isinstance(cm2_value, (int, float)):
            cm2_text = f"CM2: {cm2_value:.1f}px"
        else:
            cm2_text = f"CM2: {cm2_value}"
        cm2_color = RED if self.cm_mode == 2 else GRAY
        cm2_surface = self.font_text.render(cm2_text, True, cm2_color)
        self.screen.blit(cm2_surface, (panel_rect.x + 10, panel_rect.y + y_offset))
        y_offset += 40
        
        # R1
        r1_text = f"r1: {data.get('r1', 'N/D')}"
        if isinstance(data.get('r1'), (int, float)):
            r1_text = f"r1: {data['r1']:.3f}cm"
        r1_surface = self.font_text.render(r1_text, True, GREEN)
        self.screen.blit(r1_surface, (panel_rect.x + 10, panel_rect.y + y_offset))
        y_offset += 30
        
        R1_text = f"R1: {data.get('R1', 'N/D')}"
        if isinstance(data.get('R1'), (int, float)):
            R1_text = f"R1: {data['R1']:.3f}cm"
        R1_surface = self.font_text.render(R1_text, True, GREEN)
        self.screen.blit(R1_surface, (panel_rect.x + 10, panel_rect.y + y_offset))
        y_offset += 50
        
        # R2
        r2_text = f"r2: {data.get('r2', 'N/D')}"
        if isinstance(data.get('r2'), (int, float)):
            r2_text = f"r2: {data['r2']:.3f}cm"
        r2_surface = self.font_text.render(r2_text, True, BLUE)
        self.screen.blit(r2_surface, (panel_rect.x + 10, panel_rect.y + y_offset))
        y_offset += 30
        
        R2_text = f"R2: {data.get('R2', 'N/D')}"
        if isinstance(data.get('R2'), (int, float)):
            R2_text = f"R2: {data['R2']:.3f}cm"
        R2_surface = self.font_text.render(R2_text, True, BLUE)
        self.screen.blit(R2_surface, (panel_rect.x + 10, panel_rect.y + y_offset))
        y_offset += 60
        
        # Instruções
        instructions = [
            "Controles:",
            "← → : Mover linha amarela",
            "Enter : Medir na linha amarela",
            "PgUp/PgDn : Navegar voltagens",
            "↑ ↓ : Scroll tabela",
            "WAD : Navegar células (↓ para baixo)",
            "Clique : Definir célula WASD",
            "0 : Modo Centro (CM1/CM2)",
            "Caps : Alternar CM1 ↔ CM2",
            "1 : Modo R1",
            "2 : Modo R2",
            "S : Salvar CSV",
            "ESC : Sair"
        ]
        
        for i, instruction in enumerate(instructions):
            color = BLACK if i == 0 else GRAY
            font = self.font_text if i == 0 else self.font_small
            inst_surface = font.render(instruction, True, color)
            self.screen.blit(inst_surface, (panel_rect.x + 10, panel_rect.y + y_offset + i * 25))
        
        # Navegação entre voltagens
        nav_y = IMAGE_HEIGHT + 50
        nav_text = f"Voltagem: {self.current_voltage}V ({self.voltages.index(self.current_voltage) + 1}/{len(self.voltages)})"
        nav_surface = self.font_text.render(nav_text, True, BLACK)
        self.screen.blit(nav_surface, (50, nav_y))
        
        # Desenhar tabela CSV
        self.draw_csv_table()
    
    def run(self):
        """Loop principal da aplicação."""
        clock = pygame.time.Clock()
        running = True
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    
                    elif event.key == pygame.K_LEFT:
                        self.move_selection_line("left")
                    
                    elif event.key == pygame.K_RIGHT:
                        self.move_selection_line("right")
                    
                    elif event.key == pygame.K_PAGEUP:
                        self.change_voltage("prev")
                    
                    elif event.key == pygame.K_PAGEDOWN:
                        self.change_voltage("next")
                    
                    elif event.key == pygame.K_UP:
                        self.scroll_table("up")
                    
                    elif event.key == pygame.K_DOWN:
                        self.scroll_table("down")
                    
                    # Controles WASD para navegação de células
                    elif event.key == pygame.K_w:
                        self.navigate_table_cell("up")
                    
                    # S removido da navegação - usado só para salvar
                    
                    elif event.key == pygame.K_a:
                        self.navigate_table_cell("left")
                    
                    elif event.key == pygame.K_d:
                        self.navigate_table_cell("right")
                    
                    elif event.key == pygame.K_RETURN:
                        self.make_measurement_at_selection()
                    
                    elif event.key == pygame.K_0:
                        self.change_mode("CM")
                    
                    elif event.key == pygame.K_1:
                        self.change_mode("R1")
                    
                    elif event.key == pygame.K_2:
                        self.change_mode("R2")
                    
                    elif event.key == pygame.K_CAPSLOCK:
                        self.toggle_cm_mode()
                    
                    elif event.key == pygame.K_s:
                        self.save_csv_data()
                
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Clique esquerdo
                        self.handle_click(event.pos)
                    elif event.button == 4:  # Scroll up
                        self.scroll_table("up")
                    elif event.button == 5:  # Scroll down
                        self.scroll_table("down")
            
            self.draw_interface()
            pygame.display.flip()
            clock.tick(60)
        
        # Salvar dados automaticamente ao sair
        self.save_csv_data()
        pygame.quit()

def main():
    """Função principal."""
    print("🚀 Iniciando Análise Interativa de Difração")
    app = AnaliseDifracao()
    app.run()

if __name__ == "__main__":
    main()
