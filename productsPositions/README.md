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

#### 6. Consultar Dados
```bash
python scripts/consultar_dados.py
```
Consulta e visualiza dados do sistema:
- Posições abertas/fechadas
- Alocações
- Carteira
- Valores diários
- Resumo completo

## Estrutura

```
productsPositions/
├── domain/          # Entidades do domínio
├── services/        # Lógica de negócio
├── storage/         # Persistência (Parquet)
├── analytics/       # Consultas e análises
├── utils/           # Utilitários compartilhados
└── scripts/         # Scripts CLI
    ├── menu.py
    ├── criar_produto.py
    ├── criar_posicao.py
    ├── adicionar_stop.py
    ├── criar_alocacao.py
    ├── criar_carteira.py
    └── consultar_dados.py
```

## Fluxo Recomendado

1. **Criar produto**: `python scripts/criar_produto.py`
2. **Criar posição**: `python scripts/criar_posicao.py`
3. **Adicionar stops** (opcional): `python scripts/adicionar_stop.py`
4. **Criar alocação** (opcional): `python scripts/criar_alocacao.py`
5. **Criar carteira** (opcional): `python scripts/criar_carteira.py`
6. **Consultar dados**: `python scripts/consultar_dados.py`

## Dados

Todos os dados são armazenados em Parquet em:
```
data_parquet/products_positions/
```

## Notas

- Cada script é independente e pode ser executado separadamente
- Valores diários são importados automaticamente do CoinGecko quando `coingecko_id` é fornecido

