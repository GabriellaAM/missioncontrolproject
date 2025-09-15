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
                'aerodrome-finance', 'morpho', 'ethena', 'virtual-protocol', 'hyperliquid', 'bittensor',
                'sui']

carteira_HB = ['ethereum', 'tether', 'maker', 'havven', 'aave', 'uniswap', 'dydx',
              'cosmos', 'secret', 'the-sandbox', 'helium', 'matic-network',
              'near', 'decentraland',
              'curve-dao-token', 'gmx', 'optimism',
              'gala', 'flow', 'lido-dao', 'frax-share', 'numeraire',
              'gains-network', 'injective-protocol', 'arbitrum', 'pendle', 'radiant-capital',
              'chainlink', 'solana', 'avalanche-2', 'kujira', 'echelon-prime', 'akash-network', 
              'render-token', 'beam-2', 'multibit',
              'ondo-finance', 'ether-fi', 'aerodrome-finance', 'morpho', 
              'virtual-protocol', 'ethena', 'curve-dao-token', 'deep']

carteira_LC = ['arweave', 'badger-dao', 'my-neighbor-alice', 'perpetual-protocol',
              'alpha-finance', 'yield-guild-games', 'genopets', 'acala','rainbow-token-2',
              'guild-of-guardians', 'aurory', 'illuvium', 'conic-finance', 'vela-token',
              'radiant-capital', 'botto', 'pendle', 'nunet', 'kryptonite', 'prisma-governance-token',
              'genesysgo-shadow', 'neon', 'mintlayer', 'ethervista', 'heyanon', 'yne', 'griffain']

carteira_AC = ['bitcoin', 'ethereum', 'solana', 'maker', 'chainlink', 'thorchain', 
               'blockstack', 'immutable-x', 'uniswap', 'pendle', 'aave',
               'ethervista', 'morpho', 'ethena', 'virtual-protocol', 'yne', 'bittensor', 'sui']

others = ['fartcoin', 'raydium', 'aixbt', 'griffain', 'orbit-3', 'hyperliquid', 'yne', 'usd-coin', 'sui', 
          'hedera-hashgraph', 'mantra-dao', 'bittensor', 'pepe', 'sonic-3', 'jupiter-exchange-solana', 'dogecoin',
          'tron', 'stellar', 'chex-token', 'deep', 'instadapp', 'syrup', 'kamino', 'moonwell-artemis',
          'euler', 'grass', 'solayer', 'worldcoin-wld', 'fetch-ai', 'sei-network', 'story-2', 'aioz-network',
          'plume', 'axelar', 'kaspa', 'quant-network', 'venice-token', 'sturdy', 'verasity', 'y', 'metacade', 
          'alchemist-ai', 'anzen-finance', 'bertram-the-pomeranian', 'fwog', 'paal-ai', 'power-ledger', 'neiro-3',
          'mog-coin', 'tokenbot-2', 'cookie', 'kaito', 'eigenlayer', 'spx6900', 'grok-2', 'mamo', 'goatseus-maximus', 
          'coredaoorg', 'hypercycle', 'equilibria-finance', 'ava-ai', 'zeus-network', 'evan-2', 'alchemy-pay',
          'arbdoge-ai', 'arkham', 'arpa', 'clearpool', 'dusk-network', 'new-xai-gork', 'floki', 'jasmycoin', 'pudgy-penguins',
          'mubarak', 'bubblemaps', 'baby-doge-coin', 'chill-guy', 'liquity', 'book-of-meme', 'prometeus', 'myria',
          'slerf', 'zebec-network', 'lagrange', 'shiba-inu', 'official-trump', 'dogwifcoin', 'bonk','popcat',
          'gigachad-2', 'harrypotterobamasonic10in', 'kekius-maximus', 'based-brett', 'dog-go-to-the-moon-rune', 
          'siren-2', 'apu-s-club', 'turbo', 'cat-in-a-dogs-world', 'snek', 'ai16z', 'aethir', 'useless-3', 'constitutiondao',
          'velvet', 'bio-protocol']  

combo = carteira_AC + carteira_EXC + carteira_HB + carteira_LC + others

ROSTER = list(set(combo))