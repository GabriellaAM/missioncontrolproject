import numpy as np
import pandas as pd

def generate_hmm_states(hmm, test_processed, f_test, mode='long-only', aggressor=False):
    # Calculate Sharpe ratios for each state
    state_shares = [
        hmm.means_[i][0] / np.sqrt(np.diag(hmm.covars_[i])[0]) if np.diag(hmm.covars_[i])[0] > 0 else -np.inf 
        for i in range(3)
    ]

    # Create mapping from original to sorted states
    sorted_indices = np.argsort(state_shares)
    
    # Reorder HMM parameters
    hmm.means_ = hmm.means_[sorted_indices]
    hmm.covars_ = hmm.covars_[sorted_indices]
    hmm.transmat_ = hmm.transmat_[sorted_indices][:, sorted_indices]
    hmm.startprob_ = hmm.startprob_[sorted_indices]
    
    # Recalculate Sharpe ratios after sorting
    state_shares = [
        hmm.means_[i][0] / np.sqrt(np.diag(hmm.covars_[i])[0]) if np.diag(hmm.covars_[i])[0] > 0 else -np.inf 
        for i in range(3)
    ]

    print(f'State Sharpe Ratios: {state_shares}')

    hidden_states = hmm.predict(f_test)
    hmm_state_series = test_processed.copy()
    hmm_state_series['hidden_state'] = hidden_states

    if mode == 'long-only':
        fav_state = np.argmax(state_shares)
        hmm_state_series['state'] = np.where(hmm_state_series['hidden_state'] == fav_state, 'Long', 'Flat')
        
        p_of_states = pd.DataFrame(hmm.predict_proba(f_test))
        p_of_states_fav_unfav = pd.DataFrame({
            'fav': p_of_states[fav_state],
            'unfav': 1 - p_of_states[fav_state]
        })

    else:  # long-short
        if aggressor:
            state_signals = ['Long' if sharpe >= 0 else 'Short' for sharpe in state_shares]
        else:
            min_sharpe_idx = np.argmin(state_shares)
            state_signals = ['Long' if sharpe >= 0 else 'Short' if i == min_sharpe_idx else 'Flat'
                           for i, sharpe in enumerate(state_shares)]
            
        print(f'State Signals based on Sharpe: {state_signals}')

        hmm_state_series['state'] = hmm_state_series['hidden_state'].map(lambda x: state_signals[x])

        p_of_states = pd.DataFrame(hmm.predict_proba(f_test), index=f_test.index,
                                 columns=[f'state_{i}' for i in range(3)])
        
        long_states = [f'state_{i}' for i, signal in enumerate(state_signals) if signal == 'Long']
        p_long = p_of_states[long_states].sum(axis=1) if long_states else pd.Series(0, index=f_test.index)
        p_of_states_fav_unfav = pd.DataFrame({'fav': p_long, 'unfav': 1 - p_long}, index=f_test.index)

    hmm_state_series['return'] = hmm_state_series['log'].diff()

    return hmm_state_series, p_of_states_fav_unfav

def transact_at_open(test_processed, hmm_state_series, p_of_states_fav_unfav):
    
    df = test_processed.copy()
    p_of_states_fav_unfav.index = test_processed.index
    df = df.join(hmm_state_series[['hidden_state', 'state']]).join(p_of_states_fav_unfav['unfav'])
    df['raw_position'] = df['state'].map({'Long': 1.0, 'Short': -1.0, 'Flat': 0.0})
    df['signal'] = df['raw_position'].shift(1)
    df['next_open'] = df['Open'].shift(-1)
    df['open_return'] = (df['next_open'] / df['Open']) - 1
    df['open_return'] = df['open_return'].fillna(0)
    df['strategy_return'] = df['signal'] * df['open_return']
    df['trade'] = (df['signal'] != df['signal'].shift(1)).astype(int)
    df['cum_strategy_return'] = (1 + df['strategy_return']).cumprod() - 1
    df['cum_buy_hold_return'] = (1 + df['open_return']).cumprod() - 1
    return df
