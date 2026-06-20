import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import accuracy_score

def make_meshgrid(X, resolution=100):
    """Create a meshgrid covering the data range.

    Args:
        X: (n_samples, 2) feature array.
        resolution: number of points per axis (e.g., 100 or 200).
    Returns:
        xx, yy: meshgrid arrays of shape (resolution, resolution).
    """
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    # 使用等距的 linspace 產生指定解析度的座標
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution)
    )
    return xx, yy

def plot_decision_boundary(model, X, y, title="Decision Boundary", resolution=100):
    """Return a Plotly Figure showing data points, decision surface, support vectors.

    Args:
        model: trained sklearn SVC model.
        X, y: dataset.
        title: plot title.
        resolution: meshgrid resolution passed to make_meshgrid.
    """
    xx, yy = make_meshgrid(X, resolution=resolution)
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid)
    Z = Z.reshape(xx.shape)
    # Base contour for decision regions
    fig = go.Figure(data=go.Contour(
        x=np.arange(xx.shape[1]), y=np.arange(xx.shape[0]),
        z=Z,
        showscale=False,
        colorscale=[[0, 'rgba(255,200,200,0.3)'], [1, 'rgba(200,200,255,0.3)']],
        hoverinfo='skip',
    ))
    # Scatter of training points
    fig.add_trace(go.Scatter(
        x=X[:, 0], y=X[:, 1],
        mode='markers',
        marker=dict(color=y, colorscale='Portland', line=dict(width=1, color='black')),
        name='Data points',
    ))
    # Support vectors (if attribute exists)
    if hasattr(model, "support_vectors_"):
        sv = model.support_vectors_
        fig.add_trace(go.Scatter(
            x=sv[:, 0], y=sv[:, 1],
            mode='markers',
            marker=dict(symbol='circle-open', size=12, color='black', line=dict(width=2)),
            name='Support Vectors',
        ))
    # Linear kernel margin lines
    if model.kernel == "linear":
        # w = model.coef_[0], b = model.intercept_[0]
        w = model.coef_[0]
        b = model.intercept_[0]
        # decision line: w·x + b = 0 -> y = -(w0*x + b)/w1
        x_vals = np.array([xx.min(), xx.max()])
        y_vals = -(w[0] * x_vals + b) / w[1]
        fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode='lines', name='Decision boundary', line=dict(color='black')))
        # margins: +1/||w|| and -1/||w||
        margin = 1 / np.linalg.norm(w)
        y_up = -(w[0] * x_vals + b - margin) / w[1]
        y_down = -(w[0] * x_vals + b + margin) / w[1]
        fig.add_trace(go.Scatter(x=x_vals, y=y_up, mode='lines', name='Margin +', line=dict(dash='dash', color='gray')))
        fig.add_trace(go.Scatter(x=x_vals, y=y_down, mode='lines', name='Margin -', line=dict(dash='dash', color='gray')))
    fig.update_layout(title=title, xaxis_title='X1', yaxis_title='X2', legend=dict(itemsizing='constant'))
    return fig
