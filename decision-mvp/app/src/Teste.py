
import pandas as pd

df = pd.read_csv(r'C:\Users\monique_sandoval\decision-mvp\data\df_final.csv')
print(df['score_final'].describe())
print(df['score_final'].value_counts(bins=10))
print((df['score_final'] >= 0.6).sum(), "candidatos com score >= 0.6")
