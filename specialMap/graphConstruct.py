import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# Step 1: Read the CSV using pandas
df = pd.read_csv('./data/communityGraph/graph_Qvalue.csv')

# Create a new graph
G = nx.Graph()

for index, row in df.iterrows():
    fnid = row['fnid']
    top_2_cols = row.drop('fnid').nlargest(6).index

    for max_col in top_2_cols:
        max_value = row[max_col]
        G.add_edge(fnid, int(max_col), weight=max_value)

pos = nx.spring_layout(G)  # Positioning of nodes. Other layouts available like shell_layout, spectral_layout, etc.
nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=700, width=1.0, font_size=10)
plt.title("Visualization of the Graph")
plt.show()