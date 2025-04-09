import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

filename = "connection_matrix_pre_changes.csv"
df = pd.read_csv(filename, header=0, skiprows=21, nrows=20, index_col=0, delimiter=',') # This assumes the CSV is semicolon-separated and there's a header row and an index column
print(df)
print(f'Percent connections: {(df != 0).sum().sum() / (df.shape[0] * df.shape[1]) * 100:.2f}%')
array = df.to_numpy().T # Transpose so in csv file the sending neuron labels are on the top

G = nx.DiGraph(array)
nx.set_node_attributes(G, 'grey', 'color')

for u, v, w in G.edges.data('weight'):
    if w > 0:
        G.nodes[u]['color'] = 'blue'
        G[u][v]['color'] = 'blue'
    else:
        G.nodes[u]['color'] = 'red'
        G[u][v]['color'] = 'red'

G.nodes[0]['color'] = 'green'
node_colors = [c[1] for c in G.nodes.data('color')]
edge_colors = [c[2] for c in G.edges.data('color')]
nx.draw_circular(G, node_color=node_colors, edge_color=edge_colors, with_labels=False)
# Others include nx.draw_spring, nx.draw_shell, nx.draw

plt.show()