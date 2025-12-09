import pandas as pd
import uuid
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict

class ParquetRepo:
    """
    Repositório usando Parquet para armazenar todos os dados.
    Estrutura:
    - data_parquet/products_positions/tipos.parquet
    - data_parquet/products_positions/ativos.parquet
    - data_parquet/products_positions/produtos.parquet
    - data_parquet/products_positions/posicoes.parquet
    - data_parquet/products_positions/alocacoes.parquet
    - data_parquet/products_positions/carteiras.parquet
    - (valores diários são lidos diretamente de data_parquet/crypto_data/coingecko/{coingecko_id}/data.parquet)
    - data_parquet/products_positions/ativos_rastreados.parquet
    """
    
    def __init__(self, base_path=None):
        if base_path is None:
            # Assumir que estamos em productsPositions/storage/
            script_dir = Path(__file__).parent
            project_root = script_dir.parent.parent
            base_path = project_root / "data_parquet" / "products_positions"
        
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Caminhos dos arquivos
        self.tipos_path = self.base_path / "tipos.parquet"
        self.ativos_path = self.base_path / "ativos.parquet"
        self.produtos_path = self.base_path / "produtos.parquet"
        self.posicoes_path = self.base_path / "posicoes.parquet"
        self.alocacoes_path = self.base_path / "alocacoes.parquet"
        self.carteiras_path = self.base_path / "carteiras.parquet"
        # Valores diários são lidos diretamente de crypto_data/coingecko (não duplicados aqui)
        self.ativos_rastreados_path = self.base_path / "ativos_rastreados.parquet"
        
        # Não criar diretório automaticamente (não é mais necessário)
        
        # Inicializar arquivos vazios se não existirem
        self._inicializar_arquivos()
    
    def _inicializar_arquivos(self):
        """Cria arquivos Parquet vazios com schema se não existirem"""
        if not self.tipos_path.exists():
            pd.DataFrame(columns=['nome', 'descricao', 'data_criacao']).to_parquet(
                self.tipos_path, index=False
            )
        
        # Garantir que tipos essenciais (Perpétuos e Spot) sempre existam
        self._garantir_tipos_essenciais()
        
        if not self.ativos_path.exists():
            pd.DataFrame(columns=['nome', 'coingecko_id']).to_parquet(
                self.ativos_path, index=False
            )
        
        if not self.produtos_path.exists():
            pd.DataFrame(columns=['id', 'nome', 'data_inicio', 'tipo', 'capital_inicial']).to_parquet(
                self.produtos_path, index=False
            )
        
        if not self.posicoes_path.exists():
            pd.DataFrame(columns=[
                'id', 'produto_id', 'ativo', 'coingecko_id', 'side', 'data_entrada', 
                'preco_entrada', 'data_saida', 'preco_saida', 'status', 'stops'
            ]).to_parquet(self.posicoes_path, index=False)
        
        if not self.alocacoes_path.exists():
            pd.DataFrame(columns=[
                'id', 'produto_id', 'posicao_id', 'percentual', 
                'valor_usd', 'data', 'status'
            ]).to_parquet(self.alocacoes_path, index=False)
        
        if not self.carteiras_path.exists():
            pd.DataFrame(columns=[
                'produto_id', 'valor_disponivel', 'valor_investido',
                'pnl_nao_realizado', 'valor_total', 'data_atualizacao'
            ]).to_parquet(self.carteiras_path, index=False)
        
        if not self.ativos_rastreados_path.exists():
            pd.DataFrame(columns=[
                'ativo', 'coingecko_id', 'data_primeira_insercao', 'data_ultima_atualizacao', 
                'data_historico_inicial'
            ]).to_parquet(self.ativos_rastreados_path, index=False)
    
    def _gerar_id(self):
        """Gera um ID único"""
        return int(uuid.uuid4().int % (10 ** 10))  # ID numérico de 10 dígitos
    
    def _carregar_df(self, path):
        """Carrega DataFrame do Parquet"""
        try:
            if path.exists():
                return pd.read_parquet(path)
            return pd.DataFrame()
        except Exception as e:
            raise IOError(f"Erro ao carregar arquivo {path}: {str(e)}")
    
    def _salvar_df(self, df, path):
        """Salva DataFrame no Parquet"""
        try:
            # Garantir que o diretório existe
            path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(path, index=False, compression='snappy')
        except Exception as e:
            raise IOError(f"Erro ao salvar arquivo {path}: {str(e)}")
    
    def _validar_produto_existe(self, produto_id):
        """Valida se um produto existe"""
        produto = self.carregar_produto(produto_id)
        if produto is None:
            raise ValueError(f"Produto com ID {produto_id} não existe")
        return True
    
    def _validar_posicao_existe(self, posicao_id):
        """Valida se uma posição existe"""
        df = self._carregar_df(self.posicoes_path)
        if df.empty or posicao_id not in df['id'].values:
            raise ValueError(f"Posição com ID {posicao_id} não existe")
        return True
    
    def _validar_tipo_existe(self, nome_tipo):
        """Valida se um tipo existe"""
        df = self._carregar_df(self.tipos_path)
        if df.empty or nome_tipo not in df['nome'].values:
            raise ValueError(
                f"Tipo '{nome_tipo}' não existe. "
                f"Tipos disponíveis: {', '.join(df['nome'].tolist()) if not df.empty else 'nenhum'}"
            )
        return True
    
    def _validar_ativo_existe(self, nome_ativo):
        """Valida se um ativo existe"""
        df = self._carregar_df(self.ativos_path)
        if df.empty or nome_ativo not in df['nome'].values:
            raise ValueError(
                f"Ativo '{nome_ativo}' não existe. "
                f"Ativos disponíveis: {', '.join(df['nome'].tolist()) if not df.empty else 'nenhum'}"
            )
        return True
    
    # ========== TIPOS ==========
    
    def _garantir_tipos_essenciais(self):
        """Garante que os tipos essenciais (Perpétuos e Spot) sempre existam"""
        df = self._carregar_df(self.tipos_path)
        
        tipos_essenciais = [
            {'nome': 'Perpétuos', 'descricao': 'Contratos perpétuos de criptomoedas'},
            {'nome': 'Spot', 'descricao': 'Trading spot de criptomoedas'}
        ]
        
        tipos_existentes = set(df['nome'].values) if not df.empty else set()
        
        novos_tipos = []
        for tipo in tipos_essenciais:
            if tipo['nome'] not in tipos_existentes:
                novos_tipos.append({
                    'nome': tipo['nome'],
                    'descricao': tipo['descricao'],
                    'data_criacao': datetime.now().strftime("%Y-%m-%d")
                })
        
        if novos_tipos:
            df_novos = pd.DataFrame(novos_tipos)
            df = pd.concat([df, df_novos], ignore_index=True)
            self._salvar_df(df, self.tipos_path)
    
    def registrar_tipo(self, nome, descricao=None):
        """Registra um novo tipo (se não existir)"""
        df = self._carregar_df(self.tipos_path)
        
        # Verificar se já existe
        if not df.empty and nome in df['nome'].values:
            return nome  # Já existe, retornar
        
        # Inserir novo tipo
        novo_registro = pd.DataFrame([{
            'nome': nome,
            'descricao': descricao or '',
            'data_criacao': datetime.now().strftime("%Y-%m-%d")
        }])
        df = pd.concat([df, novo_registro], ignore_index=True)
        self._salvar_df(df, self.tipos_path)
        
        return nome
    
    def listar_tipos(self):
        """Lista todos os tipos disponíveis"""
        df = self._carregar_df(self.tipos_path)
        return df['nome'].tolist() if not df.empty else []
    
    # ========== ATIVOS ==========
    
    def registrar_ativo(self, nome, coingecko_id):
        """Registra um novo ativo (se não existir)"""
        df = self._carregar_df(self.ativos_path)
        
        # Verificar se já existe
        if not df.empty and nome in df['nome'].values:
            # Atualizar coingecko_id se fornecido e diferente
            if coingecko_id:
                mask = df['nome'] == nome
                df.loc[mask, 'coingecko_id'] = coingecko_id
                self._salvar_df(df, self.ativos_path)
            return nome
        
        # Inserir novo ativo
        novo_registro = pd.DataFrame([{
            'nome': nome,
            'coingecko_id': coingecko_id
        }])
        df = pd.concat([df, novo_registro], ignore_index=True)
        self._salvar_df(df, self.ativos_path)
        
        return nome
    
    def obter_ativo(self, nome):
        """Obtém informações de um ativo"""
        df = self._carregar_df(self.ativos_path)
        resultado = df[df['nome'] == nome]
        if not resultado.empty:
            row = resultado.iloc[0]
            return {
                'nome': row['nome'],
                'coingecko_id': row['coingecko_id'] if pd.notna(row['coingecko_id']) else None
            }
        return None
    
    def listar_ativos(self):
        """Lista todos os ativos disponíveis"""
        df = self._carregar_df(self.ativos_path)
        return df.to_dict('records') if not df.empty else []
    
    # ========== PRODUTOS ==========
    
    def salvar_produto(self, produto):
        """Salva um produto (com validação de tipo)"""
        # Validar que o tipo existe
        self._validar_tipo_existe(produto.tipo.nome)
        
        df = self._carregar_df(self.produtos_path)
        
        produto_id = self._gerar_id()
        novo_registro = pd.DataFrame([{
            'id': produto_id,
            'nome': produto.nome,
            'data_inicio': produto.data_inicio,
            'tipo': produto.tipo.nome,
            'capital_inicial': produto.capital_inicial
        }])
        
        df = pd.concat([df, novo_registro], ignore_index=True)
        self._salvar_df(df, self.produtos_path)
        
        return produto_id
    
    def carregar_produto(self, produto_id):
        """Carrega um produto por ID (retorna dict)"""
        df = self._carregar_df(self.produtos_path)
        resultado = df[df['id'] == produto_id]
        if not resultado.empty:
            return resultado.iloc[0].to_dict()
        return None
    
    def carregar_produto_objeto(self, produto_id):
        """Carrega um produto por ID e retorna objeto Produto"""
        from domain.produto import Produto
        from domain.tipo import Tipo
        
        produto_dict = self.carregar_produto(produto_id)
        if not produto_dict:
            return None
        
        tipo = Tipo(produto_dict['tipo'])
        capital_inicial = produto_dict.get('capital_inicial', 0.0)
        if pd.isna(capital_inicial):
            capital_inicial = 0.0
        
        produto = Produto(
            nome=produto_dict['nome'],
            data_inicio=produto_dict['data_inicio'],
            tipo=tipo,
            capital_inicial=float(capital_inicial)
        )
        return produto
    
    def listar_produtos(self):
        """Lista todos os produtos"""
        df = self._carregar_df(self.produtos_path)
        return df.to_dict('records') if not df.empty else []
    
    def obter_produto_por_nome(self, nome):
        """Obtém um produto por nome"""
        df = self._carregar_df(self.produtos_path)
        resultado = df[df['nome'] == nome]
        if not resultado.empty:
            return resultado.iloc[0].to_dict()
        return None
    
    # ========== POSIÇÕES ==========
    
    def salvar_posicao(self, produto_id, posicao):
        """Salva uma posição (com validação de integridade referencial)"""
        # Validar que o produto existe
        self._validar_produto_existe(produto_id)
        
        # Registrar ativo se não existir (com coingecko_id se fornecido)
        # Validar se ativo já existe e se coingecko_id é compatível
        if posicao.coingecko_id:
            ativo_existente = self.obter_ativo(posicao.ativo)
            if ativo_existente and ativo_existente.get('coingecko_id'):
                # Se já existe com coingecko_id diferente, avisar mas permitir atualização
                if ativo_existente['coingecko_id'] != posicao.coingecko_id:
                    # Atualizar com novo coingecko_id
                    self.registrar_ativo(posicao.ativo, posicao.coingecko_id)
            else:
                # Registrar novo ativo
                self.registrar_ativo(posicao.ativo, posicao.coingecko_id)
            
            # Valores diários são lidos diretamente do CoinGecko quando necessário
            # Não há necessidade de importar/duplicar dados
        
        df = self._carregar_df(self.posicoes_path)
        
        posicao_id = self._gerar_id()
        # Serializar stops como JSON string
        stops_json = json.dumps(posicao.stops) if posicao.stops else "[]"
        
        novo_registro = pd.DataFrame([{
            'id': posicao_id,
            'produto_id': produto_id,
            'ativo': posicao.ativo,
            'coingecko_id': posicao.coingecko_id if posicao.coingecko_id else None,
            'side': posicao.side,
            'data_entrada': posicao.data_entrada,
            'preco_entrada': posicao.preco_entrada,
            'data_saida': posicao.data_saida,
            'preco_saida': posicao.preco_saida,
            'status': posicao.status,
            'stops': stops_json
        }])
        
        df = pd.concat([df, novo_registro], ignore_index=True)
        self._salvar_df(df, self.posicoes_path)
        
        return posicao_id
    
    def carregar_posicoes_abertas(self, produto_id=None):
        """Carrega posições abertas"""
        df = self._carregar_df(self.posicoes_path)
        resultado = df[df['status'] == 'open']
        if produto_id:
            resultado = resultado[resultado['produto_id'] == produto_id]
        return resultado
    
    def carregar_posicoes_fechadas(self, produto_id=None):
        """Carrega posições fechadas"""
        df = self._carregar_df(self.posicoes_path)
        resultado = df[df['status'] == 'closed']
        if produto_id:
            resultado = resultado[resultado['produto_id'] == produto_id]
        return resultado
    
    def carregar_posicao(self, posicao_id):
        """Carrega uma posição por ID"""
        df = self._carregar_df(self.posicoes_path)
        resultado = df[df['id'] == posicao_id]
        if not resultado.empty:
            posicao_dict = resultado.iloc[0].to_dict()
            # Deserializar stops
            stops_json = posicao_dict.get('stops', '[]')
            try:
                posicao_dict['stops'] = json.loads(stops_json) if stops_json else []
            except (json.JSONDecodeError, TypeError):
                posicao_dict['stops'] = []
            return posicao_dict
        return None
    
    def atualizar_posicao(self, posicao_id, **kwargs):
        """Atualiza uma posição existente"""
        df = self._carregar_df(self.posicoes_path)
        
        if df.empty or posicao_id not in df['id'].values:
            raise ValueError(f"Posição com ID {posicao_id} não existe")
        
        # Atualizar campos permitidos
        campos_permitidos = ['ativo', 'coingecko_id', 'side', 'data_entrada', 
                            'preco_entrada', 'data_saida', 'preco_saida', 'status', 'stops']
        
        mask = df['id'] == posicao_id
        for campo, valor in kwargs.items():
            if campo in campos_permitidos:
                # Se for stops, serializar como JSON
                if campo == 'stops':
                    if isinstance(valor, list):
                        df.loc[mask, campo] = json.dumps(valor)
                    else:
                        df.loc[mask, campo] = valor
                else:
                    df.loc[mask, campo] = valor
            else:
                raise ValueError(f"Campo '{campo}' não é permitido para atualização")
        
        # Se atualizando coingecko_id, atualizar também na tabela de ativos
        if 'coingecko_id' in kwargs and 'ativo' in kwargs:
            self.registrar_ativo(kwargs['ativo'], kwargs['coingecko_id'])
        elif 'coingecko_id' in kwargs:
            # Obter ativo da posição
            posicao = df[df['id'] == posicao_id].iloc[0]
            if posicao['ativo']:
                self.registrar_ativo(posicao['ativo'], kwargs['coingecko_id'])
        
        self._salvar_df(df, self.posicoes_path)
        return posicao_id
    
    def adicionar_stop_posicao(self, posicao_id, data, valor):
        """
        Adiciona um novo stop a uma posição existente
        
        Args:
            posicao_id: ID da posição
            data: Data do stop (YYYY-MM-DD)
            valor: Valor do stop
        
        Returns:
            int: ID da posição atualizada
        """
        df = self._carregar_df(self.posicoes_path)
        
        if df.empty or posicao_id not in df['id'].values:
            raise ValueError(f"Posição com ID {posicao_id} não existe")
        
        mask = df['id'] == posicao_id
        posicao_row = df[mask].iloc[0]
        
        # Carregar stops existentes
        stops_json = posicao_row.get('stops', '[]')
        try:
            stops = json.loads(stops_json) if stops_json else []
        except (json.JSONDecodeError, TypeError):
            stops = []
        
        # Adicionar novo stop
        novo_stop = {"data": data, "valor": float(valor)}
        stops.append(novo_stop)
        # Ordenar por data
        stops.sort(key=lambda x: x["data"])
        
        # Atualizar
        df.loc[mask, 'stops'] = json.dumps(stops)
        self._salvar_df(df, self.posicoes_path)
        
        return posicao_id
    
    # ========== ALOCAÇÕES ==========
    
    def salvar_alocacao(self, produto_id, alocacao):
        """Salva uma alocação (com validação de integridade referencial)"""
        # Validar que o produto existe
        self._validar_produto_existe(produto_id)
        
        # Validar que a posição existe
        self._validar_posicao_existe(alocacao.posicao_id)
        
        # Validar que a posição pertence ao produto
        df_posicoes = self._carregar_df(self.posicoes_path)
        posicao = df_posicoes[
            (df_posicoes['id'] == alocacao.posicao_id) & 
            (df_posicoes['produto_id'] == produto_id)
        ]
        if posicao.empty:
            raise ValueError(
                f"Posição {alocacao.posicao_id} não pertence ao produto {produto_id}"
            )
        
        df = self._carregar_df(self.alocacoes_path)
        
        alocacao_id = self._gerar_id()
        novo_registro = pd.DataFrame([{
            'id': alocacao_id,
            'produto_id': produto_id,
            'posicao_id': alocacao.posicao_id,
            'percentual': alocacao.percentual,
            'valor_usd': alocacao.valor_usd,
            'data': alocacao.data,
            'status': alocacao.status
        }])
        
        df = pd.concat([df, novo_registro], ignore_index=True)
        self._salvar_df(df, self.alocacoes_path)
        
        return alocacao_id
    
    def salvar_alocacoes(self, produto_id, alocacoes):
        """Salva múltiplas alocações (com validação de integridade referencial)"""
        # Validar que o produto existe
        self._validar_produto_existe(produto_id)
        
        df_posicoes = self._carregar_df(self.posicoes_path)
        df = self._carregar_df(self.alocacoes_path)
        
        novos_registros = []
        ids = []
        for alocacao in alocacoes:
            # Validar que a posição existe
            self._validar_posicao_existe(alocacao.posicao_id)
            
            # Validar que a posição pertence ao produto
            posicao = df_posicoes[
                (df_posicoes['id'] == alocacao.posicao_id) & 
                (df_posicoes['produto_id'] == produto_id)
            ]
            if posicao.empty:
                raise ValueError(
                    f"Posição {alocacao.posicao_id} não pertence ao produto {produto_id}"
                )
            
            alocacao_id = self._gerar_id()
            ids.append(alocacao_id)
            novos_registros.append({
                'id': alocacao_id,
                'produto_id': produto_id,
                'posicao_id': alocacao.posicao_id,
                'percentual': alocacao.percentual,
                'valor_usd': alocacao.valor_usd,
                'data': alocacao.data,
                'status': alocacao.status
            })
        
        if novos_registros:
            df = pd.concat([df, pd.DataFrame(novos_registros)], ignore_index=True)
            self._salvar_df(df, self.alocacoes_path)
        
        return ids
    
    def carregar_alocacoes_ativas(self, produto_id):
        """Carrega alocações ativas de um produto"""
        df = self._carregar_df(self.alocacoes_path)
        resultado = df[(df['produto_id'] == produto_id) & (df['status'] == 'active')]
        return resultado
    
    def carregar_alocacao(self, alocacao_id):
        """Carrega uma alocação por ID"""
        df = self._carregar_df(self.alocacoes_path)
        resultado = df[df['id'] == alocacao_id]
        if not resultado.empty:
            return resultado.iloc[0].to_dict()
        return None
    
    def atualizar_alocacao(self, alocacao_id, **kwargs):
        """Atualiza uma alocação existente"""
        df = self._carregar_df(self.alocacoes_path)
        
        if df.empty or alocacao_id not in df['id'].values:
            raise ValueError(f"Alocação com ID {alocacao_id} não existe")
        
        # Atualizar campos permitidos
        campos_permitidos = ['percentual', 'valor_usd', 'data', 'status']
        
        mask = df['id'] == alocacao_id
        for campo, valor in kwargs.items():
            if campo in campos_permitidos:
                df.loc[mask, campo] = valor
            else:
                raise ValueError(f"Campo '{campo}' não é permitido para atualização")
        
        self._salvar_df(df, self.alocacoes_path)
        return alocacao_id
    
    # ========== CARTEIRAS ==========
    
    def salvar_carteira(self, carteira):
        """Salva ou atualiza uma carteira (UPSERT com validação de integridade)"""
        # Validar que o produto existe
        self._validar_produto_existe(carteira.produto_id)
        
        df = self._carregar_df(self.carteiras_path)
        
        # Verificar se já existe
        if not df.empty and carteira.produto_id in df['produto_id'].values:
            # Atualizar (corrigido: atualizar coluna por coluna)
            mask = df['produto_id'] == carteira.produto_id
            df.loc[mask, 'valor_disponivel'] = carteira.valor_disponivel
            df.loc[mask, 'valor_investido'] = carteira.valor_investido
            df.loc[mask, 'pnl_nao_realizado'] = carteira.pnl_nao_realizado
            df.loc[mask, 'valor_total'] = carteira.valor_total
            df.loc[mask, 'data_atualizacao'] = carteira.data_atualizacao
        else:
            # Inserir
            novo_registro = pd.DataFrame([{
                'produto_id': carteira.produto_id,
                'valor_disponivel': carteira.valor_disponivel,
                'valor_investido': carteira.valor_investido,
                'pnl_nao_realizado': carteira.pnl_nao_realizado,
                'valor_total': carteira.valor_total,
                'data_atualizacao': carteira.data_atualizacao
            }])
            df = pd.concat([df, novo_registro], ignore_index=True)
        
        self._salvar_df(df, self.carteiras_path)
        return carteira.produto_id
    
    def carregar_carteira(self, produto_id):
        """Carrega carteira de um produto"""
        from domain.carteira import Carteira
        
        df = self._carregar_df(self.carteiras_path)
        resultado = df[df['produto_id'] == produto_id]
        
        if not resultado.empty:
            row = resultado.iloc[0]
            return Carteira(
                produto_id=row['produto_id'],
                valor_disponivel=row['valor_disponivel'] or 0.0,
                valor_investido=row['valor_investido'] or 0.0,
                pnl_nao_realizado=row['pnl_nao_realizado'] or 0.0,
                valor_total=row['valor_total'],
                data_atualizacao=row['data_atualizacao']
            )
        return None
    
    # ========== VALORES DIÁRIOS ==========
    
    def verificar_ativo_rastreado(self, ativo):
        """Verifica se um ativo já está sendo rastreado"""
        df = self._carregar_df(self.ativos_rastreados_path)
        resultado = df[df['ativo'] == ativo]
        
        if not resultado.empty:
            row = resultado.iloc[0]
            return {
                'ativo': row['ativo'],
                'coingecko_id': row.get('coingecko_id') if 'coingecko_id' in row else None,
                'data_primeira_insercao': row['data_primeira_insercao'],
                'data_ultima_atualizacao': row['data_ultima_atualizacao'],
                'data_historico_inicial': row['data_historico_inicial']
            }
        return None
    
    def registrar_ativo_rastreado(self, ativo, data_historico_inicial=None, coingecko_id=None):
        """Registra um ativo como rastreado"""
        df = self._carregar_df(self.ativos_rastreados_path)
        data_atual = datetime.now().strftime("%Y-%m-%d")
        
        # Obter coingecko_id do ativo se não fornecido
        if coingecko_id is None:
            ativo_info = self.obter_ativo(ativo)
            if ativo_info:
                coingecko_id = ativo_info.get('coingecko_id')
        
        # Verificar se já existe
        if not df.empty and ativo in df['ativo'].values:
            # Atualizar
            mask = df['ativo'] == ativo
            df.loc[mask, 'data_ultima_atualizacao'] = data_atual
            if data_historico_inicial:
                df.loc[mask, 'data_historico_inicial'] = data_historico_inicial
            if coingecko_id:
                df.loc[mask, 'coingecko_id'] = coingecko_id
        else:
            # Inserir
            novo_registro = pd.DataFrame([{
                'ativo': ativo,
                'coingecko_id': coingecko_id,
                'data_primeira_insercao': data_atual,
                'data_ultima_atualizacao': data_atual,
                'data_historico_inicial': data_historico_inicial
            }])
            df = pd.concat([df, novo_registro], ignore_index=True)
        
        self._salvar_df(df, self.ativos_rastreados_path)
        return ativo
    
    def obter_valores_diarios_ativo(self, ativo, data_inicio=None, data_fim=None):
        """
        Obtém valores diários de um ativo diretamente do CoinGecko
        (não duplica dados, lê da fonte original)
        """
        # Buscar coingecko_id do ativo
        ativo_info = self.obter_ativo(ativo)
        if not ativo_info or not ativo_info.get('coingecko_id'):
            return []  # Ativo não tem coingecko_id
        
        coingecko_id = ativo_info['coingecko_id']
        
        # Usar ValorDiarioService para ler diretamente do CoinGecko
        from services.valor_diario_service import ValorDiarioService
        
        valores = ValorDiarioService.ler_valores_do_coingecko(
            coingecko_id=coingecko_id,
            data_inicio=data_inicio,
            data_fim=data_fim
        )
        
        # Converter para formato esperado (lista de tuplas)
        return [(v['data'], v['preco']) for v in valores]

