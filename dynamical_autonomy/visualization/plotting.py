"""
Plotting functions for visualizing time series and causality networks.
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
from pyvis.network import Network
from jinja2 import Template


def plot_time_series(data, fitted_data=None, title="Time Series"):
    """
    Plot the raw time series and optionally the fitted VAR model, 
    shifting each variable vertically for better visualization.

    Parameters:
    data (numpy.ndarray): Original time series, shape (n_obs, n_vars) or (n_vars, n_obs)
    fitted_data (numpy.ndarray, optional): Fitted time series from VAR model
    title (str, optional): Title of the plot
    """
    plt.figure(figsize=(10, 6))
    
    # Ensure data is in the right shape (n_obs, n_vars)
    if data.shape[0] < data.shape[1]:
        data = data.T
    
    n_vars = data.shape[1]
    time = np.arange(data.shape[0])
    vertical_shift = 2

    for i in range(n_vars):
        plt.plot(time, data[:, i] + i * vertical_shift, label=f"Variable {i + 1}", alpha=0.7)
        
        if fitted_data is not None:
            # Ensure fitted_data is in the right shape
            if fitted_data.shape[0] < fitted_data.shape[1]:
                fitted_data = fitted_data.T
                
            # Align the fitted data with the original data
            if fitted_data.shape[0] < data.shape[0]:
                data_aligned = data[-fitted_data.shape[0]:]
                time_fitted = time[-fitted_data.shape[0]:]
            else:
                data_aligned = data
                time_fitted = time
                fitted_data = fitted_data[-data.shape[0]:]
                
            plt.plot(time_fitted, fitted_data[:, i] + i * vertical_shift, '--', 
                    label=f"Fitted Variable {i + 1}")

    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Value (shifted)")
    
    # Only show legend for first few variables to avoid clutter
    handles, labels = plt.gca().get_legend_handles_labels()
    max_legend_items = min(6, len(handles))
    plt.legend(handles[:max_legend_items], labels[:max_legend_items], loc='best')
    
    plt.grid(True)
    plt.tight_layout()
    
    return plt.gcf()


def plot_granger_causality_matrix(F, labels=None, cmap='viridis', title="Granger Causality Matrix"):
    """
    Plot a heatmap of the Granger causality matrix.

    Parameters:
    F (numpy.ndarray): Granger causality matrix, shape (n, n)
    labels (list, optional): Labels for variables
    cmap (str, optional): Colormap for the heatmap
    title (str, optional): Title of the plot

    Returns:
    matplotlib.figure.Figure: The figure object
    """
    plt.figure(figsize=(8, 6))
    
    if labels is None:
        n = F.shape[0]
        labels = [f"Var {i+1}" for i in range(n)]
    
    # Create mask for diagonal elements (self-causality is often NaN)
    mask = np.eye(F.shape[0], dtype=bool)
    
    # Plot heatmap
    ax = sns.heatmap(F, annot=True, cmap=cmap, mask=mask, 
                    xticklabels=labels, yticklabels=labels, 
                    cbar_kws={'label': 'Granger Causality (nats)'})
    
    plt.title(title)
    plt.xlabel("Source")
    plt.ylabel("Target")
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    
    return plt.gcf()


def plot_granger_causality_network(F, labels=None, threshold=0.1, title="Granger Causality Network", 
                                  html_output=None, height="500px", width="100%", 
                                  node_color="#97c2fc", edge_color_pos="#2B7CE9"):
    """
    Plot a network visualization of Granger causality relationships.

    Parameters:
    F (numpy.ndarray): Granger causality matrix, shape (n, n)
    labels (list, optional): Labels for variables
    threshold (float, optional): Threshold for including edges
    title (str, optional): Title of the plot
    html_output (str, optional): Path to save HTML network visualization
    height (str, optional): Height of the HTML visualization
    width (str, optional): Width of the HTML visualization
    node_color (str, optional): Color for nodes
    edge_color_pos (str, optional): Color for positive edges

    Returns:
    tuple: (NetworkX graph, PyVis network)
    """
    n = F.shape[0]
    
    if labels is None:
        labels = [f"Var {i+1}" for i in range(n)]
    
    # Create NetworkX graph
    G = nx.DiGraph()
    
    # Add nodes with labels
    for i in range(n):
        G.add_node(i, label=labels[i])
    
    # Add edges for Granger causality above threshold
    for i in range(n):
        for j in range(n):
            # Skip diagonal (self-causality) and values below threshold or NaN
            if i != j and not np.isnan(F[i, j]) and F[i, j] > threshold:
                G.add_edge(j, i, weight=F[i, j], title=f"GC: {F[i, j]:.3f}")
    
    # Create PyVis network for interactive visualization
    net = Network(notebook=False, directed=True, height=height, width=width)
    
    # Add nodes
    for node, node_attrs in G.nodes(data=True):
        net.add_node(node, label=node_attrs.get('label', str(node)), 
                   title=f"Node: {node_attrs.get('label', str(node))}", 
                   color=node_color)
    
    # Add edges with weights and tooltips
    for source, target, edge_attrs in G.edges(data=True):
        weight = edge_attrs.get('weight', 1.0)
        # Scale edge width based on weight
        width = 1 + 5 * weight / (F.max() or 1)
        net.add_edge(source, target, value=width, title=edge_attrs.get('title', ''),
                   color=edge_color_pos)
    
    # Apply physics and other options
    net.toggle_physics(True)
    net.set_options("""
    {
      "physics": {
        "barnesHut": {
          "gravitationalConstant": -2000,
          "centralGravity": 0.3,
          "springLength": 150,
          "springConstant": 0.04
        },
        "minVelocity": 0.75
      }
    }
    """)
    
    # Save to HTML if requested
    if html_output:
        # Add a custom title to the HTML output
        html_template = Template("""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>{{ title }}</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    margin: 0;
                    padding: 20px;
                }
                h1 {
                    text-align: center;
                    color: #333;
                }
                #visualization {
                    border: 1px solid #ddd;
                    border-radius: 5px;
                }
            </style>
        </head>
        <body>
            <h1>{{ title }}</h1>
            {{ graph_html }}
        </body>
        </html>
        """)
        
        # Save to a temporary file first
        net.save_graph("temp_network.html")
        
        # Read the content
        with open("temp_network.html", "r") as f:
            graph_html = f.read()
        
        # Render the template
        html_content = html_template.render(title=title, graph_html=graph_html)
        
        # Save to the final file
        with open(html_output, "w") as f:
            f.write(html_content)
    
    return G, net 