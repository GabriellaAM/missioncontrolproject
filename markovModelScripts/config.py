MACRO_FILES = ['creditSpreads', 'treasury5YInflationExpectation', 'move', 'financialConditionsIndex']

FEATURES_BTC = ['log_return_30', 'log_return_7', 'RSI14', 'ln_atr', 'kalman_smooth_slope',
                'garch_volatility', 'volume_return', 'move_log_return', 'treasury5YInflationExpectation_log_return']

FEATURES_ALTS = ['kalman_smooth_slope', 'ln_atr', 'log_return_30', 'log_return_7', 'RSI14',
                 'garch_volatility', 'volume_return', 'btc_dom_return_7']

RANDOM_SEED = 4214609