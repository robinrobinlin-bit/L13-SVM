import numpy as np
import plotly.graph_objects as go
from typing import Tuple, Dict

def scatter_data(X: np.ndarray, y: np.ndarray, title: str = "Data Points") -> go.Figure:
    """Return a Plotly scatter figure for the dataset.

    Parameters
    ----------
    X: (n_samples, 2) feature matrix
    y: (n_samples,) label vector (0/1)
    title: chart title
    """
    fig = go.Figure()
    colors = ["#1f77b4", "#ff7f0e"]  # blue, orange/red
    for label in np.unique(y):
        mask = y == label
        fig.add_trace(
            go.Scatter(
                x=X[mask, 0],
                y=X[mask, 1],
                mode="markers",
                marker=dict(color=colors[int(label)], size=8),
                name=f"Class {label}",
            )
        )
    fig.update_layout(title=title, legend=dict(itemsizing="constant"), height=500, width=700)
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    return fig

def decision_boundary(
    xx: np.ndarray,
    yy: np.ndarray,
    Z: np.ndarray,
    title: str = "Decision Boundary",
) -> go.Figure:
    """Create a contour plot of the decision surface.

    Parameters
    ----------
    xx, yy: meshgrid arrays
    Z: predicted label for each point (same shape as xx)
    """
    fig = go.Figure(data=go.Contour(
        x=xx[0],
        y=yy[:, 0],
        z=Z,
        colorscale=[[0, "#1f77b4"], [1, "#ff7f0e"]],
        showscale=False,
        opacity=0.5,
    ))
    fig.update_layout(title=title, height=500, width=700)
    return fig

def plot_3d_mapping(
    X: np.ndarray,
    Z: np.ndarray,
    title: str = "3D Kernel Mapping",
) -> go.Figure:
    """Plot a 3‑D scatter where Z is the third dimension.
    """
    fig = go.Figure(data=go.Scatter3d(
        x=X[:, 0],
        y=X[:, 1],
        z=Z,
        mode="markers",
        marker=dict(size=5, color=Z, colorscale="Viridis"),
    ))
    fig.update_layout(title=title, scene=dict(xaxis_title="x1", yaxis_title="x2", zaxis_title="z"))
    return fig
