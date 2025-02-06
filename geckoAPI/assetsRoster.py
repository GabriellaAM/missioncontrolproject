carteira_EXC = ['aave', 'arbitrum', 'arweave','audius','aurory','avalanche-2','axie-infinity',
                'balancer','band-protocol','binancecoin','bitcoin-cash','cardano',
                'eos','litecoin','polkadot','ripple','bitcoin','curve-dao-token','tether',
                'true-usd','chainlink','compound-governance-token','cosmos','crypto-com-chain',
                'dash','decentraland', 'dodo', 'dydx','ethereum', 'ethereum-classic', 'fantom', 
                'flow', 'gmx', 'stellar', 'havven', 'helium', 'internet-computer',
                'kyber-network-crystal', 'lido-dao','lisk', 'maker', 'matic-network', 'monero', 
                'polymath', 'near', 'omisego', 'optimism', 'pancakeswap-token', 'pax-gold',
                'pendle', 'star-atlas-dao', 'ronin', 'rari-governance-token', 'the-sandbox', 
                'secret', 'serum', 'smartcash', 'solana', 'spell-token', 'sushi', 'swarm',
                'terra-luna-2', 'tezos', 'tribe-2', 'uniswap', 'wibx', 'wrapped-nxm', 'yearn-finance', 
                'zcash', 'frax-share', 'celestia', 'ronin', 'thorchain', 'immutable-x',
                'akash-network', 'render-token', 'blockstack', 'ondo-finance', 'the-open-network', 
                'aerodrome-finance', 'morpho', 'ethena']

carteira_HB = ['ethereum', 'tether', 'maker', 'havven', 'aave', 'uniswap', 'dydx',
              'cosmos', 'secret', 'the-sandbox', 'helium', 'matic-network',
              'near', 'decentraland',
              'curve-dao-token', 'gmx', 'optimism',
              'gala', 'flow', 'lido-dao', 'frax-share', 'numeraire',
              'gains-network', 'injective-protocol', 'arbitrum', 'pendle', 'radiant-capital',
              'chainlink', 'solana', 'avalanche-2', 'kujira', 'echelon-prime', 'akash-network', 
              'render-token', 'beam-2', 'multibit',
              'ondo-finance', 'ether-fi', 'aerodrome-finance', 'morpho', 
              'virtual-protocol', 'ethena']

carteira_LC = ['arweave', 'badger-dao', 'my-neighbor-alice', 'perpetual-protocol',
              'alpha-finance', 'yield-guild-games', 'genopets', 'acala','rainbow-token-2',
              'guild-of-guardians', 'aurory', 'illuvium', 'conic-finance', 'vela-token',
              'radiant-capital', 'botto', 'pendle', 'nunet', 'kryptonite', 'prisma-governance-token',
              'genesysgo-shadow', 'neon', 'mintlayer', 'ethervista', 'heyanon']

carteira_AC = ['bitcoin', 'ethereum', 'solana', 'maker', 'chainlink', 'thorchain', 
               'blockstack', 'immutable-x', 'uniswap', 'pendle', 'aave',
               'ethervista', 'morpho', 'ethena', 'virtual-protocol']

others = ['fartcoin', 'raydium', 'aixbt', 'griffain', 'orbit-3', 'hyperliquid', 'yne']

combo = carteira_AC + carteira_EXC + carteira_HB + carteira_LC + others

ROSTER = list(set(combo))