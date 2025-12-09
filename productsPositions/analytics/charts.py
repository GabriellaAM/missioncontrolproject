import plotly.express as px

# Gera gráfico de linha dos valores diários
def grafico_valores(df):
    fig = px.line(df, x="data", y="preco", title="Evolução da Posição")
    fig.show()
