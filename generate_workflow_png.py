import matplotlib.pyplot as plt

# Define components and positions
components = {
    "GitHub": (0, 2),
    "Streamlit Cloud": (2, 2),
    "Streamlit UI": (4, 2),
    "Manim Videos": (5, 1),
    "Plotly Interactive": (5, 3),
    "scikit-learn": (3, 0),
}

fig, ax = plt.subplots(figsize=(8, 4))
ax.set_axis_off()

# Draw boxes for each component
for name, (x, y) in components.items():
    ax.text(x, y, name, fontsize=12, ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.3', fc='lightblue', ec='navy'))

# Draw arrows to show flow
# GitHub -> Streamlit Cloud
ax.annotate('', xy=components["Streamlit Cloud"], xytext=components["GitHub"],
            arrowprops=dict(arrowstyle='->', color='red'))
# Streamlit Cloud -> Streamlit UI
ax.annotate('', xy=components["Streamlit UI"], xytext=components["Streamlit Cloud"],
            arrowprops=dict(arrowstyle='->', color='red'))
# Streamlit UI -> Manim Videos
ax.annotate('', xy=components["Manim Videos"], xytext=components["Streamlit UI"],
            arrowprops=dict(arrowstyle='->', color='red'))
# Streamlit UI -> Plotly Interactive
ax.annotate('', xy=components["Plotly Interactive"], xytext=components["Streamlit UI"],
            arrowprops=dict(arrowstyle='->', color='red'))
# scikit-learn -> Plotly Interactive
ax.annotate('', xy=components["Plotly Interactive"], xytext=components["scikit-learn"],
            arrowprops=dict(arrowstyle='->', color='red'))
# scikit-learn -> Streamlit UI (model training)
ax.annotate('', xy=components["Streamlit UI"], xytext=components["scikit-learn"],
            arrowprops=dict(arrowstyle='->', color='red'))

plt.tight_layout()
plt.savefig('workflow.png')
print('workflow.png saved')
