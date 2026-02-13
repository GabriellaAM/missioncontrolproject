import sqlite3
conn = sqlite3.connect('productsPositions/data/products_positions.db')
cur = conn.cursor()

# Check which product 2150859854 is
cur.execute("SELECT id, nome FROM produtos WHERE id = 2150859854")
row = cur.fetchone()
print(f"Produto 2150859854: {row}")

# Check closed positions for this product
cur.execute("""
    SELECT id, ativo, data_saida, preco_saida, status 
    FROM posicoes 
    WHERE produto_id = 2150859854 AND status = 'closed'
    ORDER BY data_saida DESC
    LIMIT 15
""")
print(f"\nPosições fechadas do produto 2150859854:")
for r in cur.fetchall():
    print(f"  id={r[0]} ativo={r[1]} data_saida={r[2]} preco_saida={r[3]} status={r[4]}")

# Count
cur.execute("SELECT COUNT(*) FROM posicoes WHERE produto_id = 2150859854 AND status='closed' AND preco_saida IS NOT NULL AND preco_saida != 0")
print(f"\nCom preco_saida: {cur.fetchone()[0]}")
cur.execute("SELECT COUNT(*) FROM posicoes WHERE produto_id = 2150859854 AND status='closed' AND (preco_saida IS NULL OR preco_saida = 0)")
print(f"Sem preco_saida: {cur.fetchone()[0]}")

# Also check all 4 special products
cur.execute("SELECT id, nome FROM produtos WHERE nome IN ('Alphacoins', 'EXC', 'HB', 'LC')")
print(f"\nProdutos especiais:")
for r in cur.fetchall():
    cur2 = conn.cursor()
    cur2.execute("SELECT COUNT(*) FROM posicoes WHERE produto_id = ? AND status='closed' AND (preco_saida IS NULL OR preco_saida = 0)", (r[0],))
    sem = cur2.fetchone()[0]
    cur2.execute("SELECT COUNT(*) FROM posicoes WHERE produto_id = ? AND status='closed' AND preco_saida IS NOT NULL AND preco_saida != 0", (r[0],))
    com = cur2.fetchone()[0]
    print(f"  {r[0]} {r[1]}: com={com} sem={sem}")

conn.close()
