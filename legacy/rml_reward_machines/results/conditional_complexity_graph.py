import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

M_values = range(1, 1000)  

data = pd.DataFrame({
    'M': list(M_values) * 3,
    'Branches': list(M_values) * 2 + [1]*len(M_values),
    'Method': ['CRA']*len(M_values) + ['Reward Machine']*len(M_values) + ['RML']*len(M_values)
})

sns.set_theme(style="whitegrid")

plt.figure(figsize=(8, 6))
palette = {
    'CRA': '#1f77b4',
    'Reward Machine': '#ff7f0e', 
    'RML': '#2ca02c'
}
linestyles = {
    'CRA': (1,2),
    'Reward Machine': (5,5),
    'RML': (None,None)
}

sns.lineplot(
    data=data,
    x='M',
    y='Branches',
    hue='Method',
    style='Method',
    palette=palette,
    dashes=linestyles,
    linewidth=2.0
)

plt.xlabel('M')
plt.ylabel('Number of Branches')

plt.show()
