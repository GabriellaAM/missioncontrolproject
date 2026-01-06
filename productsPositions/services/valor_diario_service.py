from pathlib import Path
import pandas as pd
import requests
import os
from dotenv import load_dotenv

# Carregar variáveis de ambiente
load_dotenv()

# Serviço responsável por gerenciar valores diários de ativos
class ValorDiarioService:
    
    @staticmethod
    def _obter_caminho_coingecko(coingecko_id):
        """Obtém o caminho do arquivo Parquet do CoinGecko"""
        # Assumir que estamos em productsPositions/
        script_dir = Path(__file__).parent
        project_root = script_dir.parent.parent
        caminho = project_root / "data_parquet" / "crypto_data" / "coingecko" / coingecko_id / "data.parquet"
        return caminho
    
    @staticmethod
    def ler_valores_do_coingecko(coingecko_id, data_inicio=None, data_fim=None):
        """
        Lê valores diários do diretório CoinGecko
        
        Args:
            coingecko_id: ID do CoinGecko (ex: "bitcoin")
            data_inicio: Data inicial (YYYY-MM-DD) - opcional
            data_fim: Data final (YYYY-MM-DD) - opcional
        
        Returns:
            list: Lista de dicionários [{"data": "2025-01-10", "preco": 100000}, ...]
        """
        caminho = ValorDiarioService._obter_caminho_coingecko(coingecko_id)
        
        if not caminho.exists():
            return []
        
        try:
            df = pd.read_parquet(caminho)
            
            # Converter timestamp para string de data
            if 'timestamp' in df.columns:
                # Converter para datetime se necessário e depois para string
                df['data'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%d')
            elif 'date' in df.columns:
                df['data'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
            else:
                # Tentar usar o índice se for datetime
                if isinstance(df.index, pd.DatetimeIndex):
                    df['data'] = df.index.strftime('%Y-%m-%d')
                else:
                    raise ValueError("Não foi possível encontrar coluna de data no arquivo")
            
            # Usar 'close' como preço (ou 'open' se não houver close)
            if 'close' in df.columns:
                df['preco'] = df['close']
            elif 'open' in df.columns:
                df['preco'] = df['open']
            else:
                raise ValueError("Não foi possível encontrar coluna de preço no arquivo")
            
            # Filtrar por data se fornecido
            if data_inicio:
                df = df[df['data'] >= data_inicio]
            if data_fim:
                df = df[df['data'] <= data_fim]
            
            # Ordenar por data
            df = df.sort_values('data')
            
            # Converter para lista de dicionários
            valores = []
            for _, row in df.iterrows():
                valores.append({
                    'data': str(row['data']),
                    'preco': float(row['preco'])
                })
            
            return valores
            
        except Exception as e:
            # Retornar lista vazia em caso de erro (não quebrar o fluxo)
            return []
    
    @staticmethod
    def obter_preco_atual(coingecko_id):
        """
        Obtém o preço atual de um ativo via API do CoinGecko
        
        Args:
            coingecko_id: ID do CoinGecko (ex: "bitcoin")
        
        Returns:
            float ou None: Preço atual em USD, ou None se não conseguir buscar
        """
        if not coingecko_id:
            return None
        
        api_key = os.getenv('GECKO_API_KEY')
        if not api_key:
            # Tentar buscar do arquivo Parquet como fallback
            valores = ValorDiarioService.ler_valores_do_coingecko(coingecko_id)
            if valores:
                return valores[-1]['preco']  # Último preço disponível
            return None
        
        try:
            url = 'https://pro-api.coingecko.com/api/v3/simple/price'
            params = {
                'vs_currencies': 'usd',
                'ids': coingecko_id
            }
            headers = {
                'x-cg-pro-api-key': api_key
            }
            
            response = requests.get(url, params=params, headers=headers, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            if coingecko_id in data and 'usd' in data[coingecko_id]:
                return float(data[coingecko_id]['usd'])
            
            return None
            
        except requests.exceptions.RequestException:
            # Se falhar a API, tentar buscar do Parquet como fallback
            valores = ValorDiarioService.ler_valores_do_coingecko(coingecko_id)
            if valores:
                return valores[-1]['preco']  # Último preço disponível
            return None
        except Exception:
            return None

    @staticmethod
    def obter_precos_batch(coingecko_ids):
        """
        Obtém preços atuais de múltiplos ativos em uma única chamada à API do CoinGecko.

        Args:
            coingecko_ids: Lista de IDs do CoinGecko (ex: ["bitcoin", "ethereum"])

        Returns:
            dict: Mapeamento de coingecko_id -> preço em USD
        """
        if not coingecko_ids:
            return {}

        # Remove duplicates and None values
        unique_ids = list(set(cid for cid in coingecko_ids if cid))
        if not unique_ids:
            return {}

        api_key = os.getenv('GECKO_API_KEY')
        prices = {}

        if api_key:
            try:
                url = 'https://pro-api.coingecko.com/api/v3/simple/price'
                # CoinGecko accepts comma-separated IDs
                params = {
                    'vs_currencies': 'usd',
                    'ids': ','.join(unique_ids)
                }
                headers = {
                    'x-cg-pro-api-key': api_key
                }

                response = requests.get(url, params=params, headers=headers, timeout=15)
                response.raise_for_status()

                data = response.json()
                for cid in unique_ids:
                    if cid in data and 'usd' in data[cid]:
                        prices[cid] = float(data[cid]['usd'])
                    else:
                        prices[cid] = None

                return prices

            except Exception:
                # Fall through to parquet fallback
                pass

        # Fallback: read from parquet files
        for cid in unique_ids:
            try:
                valores = ValorDiarioService.ler_valores_do_coingecko(cid)
                if valores:
                    prices[cid] = valores[-1]['preco']
                else:
                    prices[cid] = None
            except Exception:
                prices[cid] = None

        return prices

