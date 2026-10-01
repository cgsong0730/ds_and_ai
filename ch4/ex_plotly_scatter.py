import plotly.express as px

df = px.data.iris()
fig = px.scatter(df, x='petal_length', y='petal_width', color='species')
fig.show()  # 브라우저에서 확대·이동 가능
