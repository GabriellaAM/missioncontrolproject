"""
Simple notebook setup functions for Soros system
"""
import warnings
from .core.portfolio_analyzer import PortfolioAnalyzer
from .config import load_config

warnings.filterwarnings('ignore')


def setup_analyzer(config_path="soros_config.yaml", asset_ids=None):
    """
    Simple function to set up PortfolioAnalyzer with configuration
    
    Args:
        config_path: Path to YAML config file
        asset_ids: List of asset IDs to analyze (optional)
    
    Returns:
        PortfolioAnalyzer instance ready to use
    """
    # Load configuration
    config = load_config(config_path)
    paths = config.get_data_paths()
    
    # Create analyzer with configured paths
    analyzer = PortfolioAnalyzer(
        data_dir=paths['base_dir'],
        data_path=paths['candle_data_path'],
        btc_data_path=paths['btc_data_path'],
        ssr_data_path=paths['ssr_data_path'],
        market_data_path=paths['market_data_path'],
        asset_ids=asset_ids
    )
    
    return analyzer


def get_portfolio_assets():
    """
    Get predefined portfolio asset lists
    
    Returns:
        Dictionary with portfolio names as keys and asset lists as values
    """
    # Import from scripts.assetsRoster if available, otherwise use defaults
    try:
        from scripts.assetsRoster import carteira_AC, carteira_HB, carteira_EXC, carteira_LC, others
        return {
            'carteira_AC': carteira_AC,
            'carteira_HB': carteira_HB,
            'carteira_EXC': carteira_EXC,
            'carteira_LC': carteira_LC,
            'others': others,
            'all': list(set(carteira_AC + carteira_HB + carteira_EXC + carteira_LC + others))
        }
    except ImportError:
        # Fallback to basic portfolios
        return {
            'basic': ['bitcoin', 'ethereum', 'solana', 'chainlink', 'uniswap'],
            'all': ['bitcoin', 'ethereum', 'solana', 'chainlink', 'uniswap']
        }


def quick_setup(portfolio='all', config_path="soros_config.yaml"):
    """
    Quick setup function for notebooks - load analyzer and data in one call
    
    Args:
        portfolio: Portfolio name ('all', 'carteira_AC', etc.) or list of asset IDs
        config_path: Path to YAML config file
    
    Returns:
        Configured PortfolioAnalyzer with data loaded
    """
    # Get assets
    if isinstance(portfolio, str):
        portfolios = get_portfolio_assets()
        assets = portfolios.get(portfolio, portfolios['all'])
    else:
        assets = portfolio
    
    # Setup analyzer
    analyzer = setup_analyzer(config_path, assets)
    
    # Load data
    print(f"Loading data for {len(assets)} assets...")
    analyzer.load_data(assets=assets, start_date='2014-01-01', end_date='2025-12-31')
    
    print(f"✓ Analyzer ready with {len(analyzer.assets)} assets loaded")
    return analyzer