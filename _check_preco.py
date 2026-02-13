import sqlite3
conn = sqlite3.connect('productsPositions/data/products_positions.db')
cur = conn.cursor()

cur.execute("SELECT id, ativo, data_saida, preco_saida, status FROM posicoes WHERE status='closed' AND preco_saida IS NOT NULL AND preco_saida != 0 LIMIT 10")
print('=== COM preco_saida ===')
for r in cur.fetchall():
    print(r)

cur.execute("SELECT id, ativo, data_saida, preco_saida, status FROM posicoes WHERE status='closed' AND (preco_saida IS NULL OR preco_saida = 0) LIMIT 15")
print('\n=== SEM preco_saida ===')
for r in cur.fetchall():
    print(r)

cur.execute("SELECT COUNT(*) FROM posicoes WHERE status='closed' AND preco_saida IS NOT NULL AND preco_saida != 0")
print(f'\nTotal com preco_saida: {cur.fetchone()[0]}')

cur.execute("SELECT COUNT(*) FROM posicoes WHERE status='closed' AND (preco_saida IS NULL OR preco_saida = 0)")
print(f'Total sem preco_saida: {cur.fetchone()[0]}')

# Check total closed positions
cur.execute("SELECT COUNT(*) FROM posicoes WHERE status='closed'")
print(f'Total posições fechadas: {cur.fetchone()[0]}')

conn.close()
