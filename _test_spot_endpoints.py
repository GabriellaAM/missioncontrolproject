"""Test: verificar novos endpoints de copy trading spot"""
import sys
sys.path.insert(0, 'productsPositions')

from services.bitget_service import (
    get_bitget_credentials,
    fetch_spot_positions,
    fetch_spot_copy_history,
    _timestamp_to_date_str,
)

def test_product(nome):
    print(f"\n{'='*60}")
    print(f"  {nome}")
    print(f"{'='*60}")
    
    creds = get_bitget_credentials(nome)
    if not creds:
        print(f"  SEM CREDENCIAIS para '{nome}'")
        return

    # Test open positions
    print(f"\n--- Posicoes abertas (order-current-track) ---")
    try:
        positions = fetch_spot_positions(creds)
        print(f"Total: {len(positions)}")
        for p in positions:
            date_str = _timestamp_to_date_str(p.get('ctime')) or '?'
            print(f"  {p['symbol']:12s}  qty={p['quantity']:>12.6f}  "
                  f"entry=${p['entry_price']:>10.4f}  "
                  f"date={date_str}  "
                  f"pnl={p.get('unrealized_pnl', 0):>8.2f}")
    except Exception as e:
        print(f"  ERRO: {e}")

    # Test history
    print(f"\n--- Historico fechadas (order-history-track) ---")
    try:
        history = fetch_spot_copy_history(creds, limit=10)
        print(f"Total (max 10): {len(history)}")
        for h in history:
            open_date = _timestamp_to_date_str(h.get('open_time')) or '?'
            close_date = _timestamp_to_date_str(h.get('close_time')) or '?'
            print(f"  {h['symbol']:12s}  qty={h['quantity']:>12.6f}  "
                  f"buy=${h['open_price']:>10.4f}  sell=${h['close_price']:>10.4f}  "
                  f"pnl={h['pnl']:>8.4f}  "
                  f"{open_date} -> {close_date}")
    except Exception as e:
        print(f"  ERRO: {e}")


if __name__ == '__main__':
    test_product('Soros Spot 1')
    test_product('Soros Spot 2')
    print("\nDone!")
