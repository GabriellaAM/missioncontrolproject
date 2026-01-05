# Sistema de Gestão de Posições e Produtos

Sistema modular para gerenciar produtos, posições, alocações e carteiras de investimento.

## Scripts Disponíveis

### Menu Interativo
```bash
python scripts/menu.py
```
Menu principal com todas as opções disponíveis.

### Scripts Individuais

#### 1. Criar Produto
```bash
python scripts/criar_produto.py
```
Cria um novo produto (fundo, carteira, etc.).

#### 2. Criar Posição
```bash
python scripts/criar_posicao.py
```
Cria uma nova posição (long/short) em um produto existente.
- Importa automaticamente valores diários do CoinGecko se `coingecko_id` for fornecido

#### 3. Adicionar Stop
```bash
python scripts/adicionar_stop.py
```
Adiciona um stop a uma posição existente (aberta ou fechada).

#### 4. Criar Alocação
```bash
python scripts/criar_alocacao.py
```
Cria uma alocação de capital em uma posição.

#### 5. Criar/Atualizar Carteira
```bash
python scripts/criar_carteira.py
```
Cria ou atualiza a carteira de um produto com 3 opções:
1. Manual (valores diretos)
2. Automática (calcula a partir das posições)
3. Com capital inicial (recalcula tudo)

#### 6. Visualizar Dados (CLI)
```bash
python scripts/visualizar_dados.py
```
Visualiza dados do sistema com tabelas formatadas e gráficos interativos:
- Dados formatados (produtos, posições, carteira, alocações)
- Gráficos interativos (evolução de preços, composição da carteira, etc.)
- Dashboard completo

#### 7. Dashboard Web
```bash
python scripts/servidor_dashboard.py
```
Interface web para visualizar todos os produtos e posições no navegador:
- Dashboard principal com cards de todos os produtos
- Visualizacao detalhada por produto
- Posicoes abertas e fechadas
- Atualizacao de dados com um clique
- Auto-refresh a cada 5 minutos

O navegador abre automaticamente em `http://localhost:8080`

**Rotas disponiveis:**
| Rota | Descricao |
|------|-----------|
| `/` | Dashboard principal |
| `/produto/{id}` | Detalhes do produto |
| `/produto/{id}/abertas` | Posicoes abertas |
| `/produto/{id}/fechadas` | Posicoes fechadas |
| `/api/produtos` | Lista produtos (JSON) |
| `/api/atualizar/{id}` | Atualizar dados |

## Estrutura

```
productsPositions/
├── domain/          # Entidades do dominio
├── services/        # Logica de negocio
├── storage/         # Persistencia (SQLite)
├── analytics/       # Consultas e analises
├── utils/           # Utilitarios compartilhados
└── scripts/         # Scripts CLI
    ├── menu.py              # Menu principal
    ├── criar_produto.py
    ├── editar_produto.py
    ├── deletar_produto.py
    ├── criar_posicao.py
    ├── editar_posicao.py
    ├── deletar_posicao.py
    ├── adicionar_stop.py
    ├── configurar_atr_stop.py
    ├── atualizar_stops_atr.py
    ├── criar_alocacao.py
    ├── criar_carteira.py
    ├── visualizar_dados.py
    ├── gerenciar_atributos.py
    ├── gerenciar_visualizacoes.py
    ├── servidor_atualizacao.py  # Servidor para botao Atualizar
    └── servidor_dashboard.py    # Dashboard Web
```

## Fluxo Recomendado

1. **Criar produto**: `python scripts/criar_produto.py`
2. **Criar posição**: `python scripts/criar_posicao.py`
3. **Adicionar stops** (opcional): `python scripts/adicionar_stop.py`
4. **Criar alocação** (opcional): `python scripts/criar_alocacao.py`
5. **Criar carteira** (opcional): `python scripts/criar_carteira.py`
6. **Visualizar dados**: `python scripts/visualizar_dados.py`

## Dados

Todos os dados são armazenados em SQLite em:
```
productsPositions/data/products_positions.db
```

**Nota:** Os valores diários de preços continuam sendo lidos de `data_parquet/crypto_data/coingecko/{coingecko_id}/data.parquet` (não são duplicados no SQLite).

## Análise em Notebooks Jupyter

O sistema inclui utilitários para análise visual em notebooks Jupyter.

### Instalação de Dependências

```bash
pip install pandas plotly jupyter
```

### Uso Básico

```python
import sys
from pathlib import Path

# Adicionar ao path
sys.path.insert(0, str(Path.cwd() / 'productsPositions'))

from analytics.notebook_utils import *
from analytics.charts import *

# Listar produtos
produtos = display_produtos()

# Visualizar posições abertas
posicoes = display_posicoes_abertas(produto_id=1)

# Visualizar carteira
carteira = display_carteira(produto_id=1)

# Gráfico de evolução de preço
df_valores = get_valores_ativo("BTC")
fig = grafico_valores(df_valores, ativo="BTC")
fig.show()

# Dashboard completo
fig = dashboard_produto(produto_id=1)
fig.show()
```

### Funções Disponíveis

**notebook_utils.py:**
- `display_produtos()` - Lista todos os produtos
- `display_posicoes_abertas(produto_id)` - Posições abertas formatadas
- `display_carteira(produto_id)` - Carteira formatada
- `display_alocacoes(produto_id)` - Alocações formatadas
- `display_resumo_completo(produto_id)` - Resumo completo
- `get_valores_ativo(ativo)` - Valores diários de um ativo
- `get_valores_posicao(posicao_id)` - Valores diários de uma posição

**charts.py:**
- `grafico_valores(df, ativo)` - Gráfico de linha de evolução de preço
- `grafico_carteira(produto_id)` - Gráfico de barras da carteira
- `grafico_alocacoes(produto_id)` - Gráfico de pizza das alocações
- `grafico_comparativo_ativos(ativos)` - Comparativo de múltiplos ativos
- `dashboard_produto(produto_id)` - Dashboard completo com múltiplos gráficos


## Notas

- Cada script é independente e pode ser executado separadamente
- Valores diários são lidos automaticamente do CoinGecko quando `coingecko_id` está disponível
- Todas as queries retornam DataFrames do pandas, prontos para análise em notebooks

