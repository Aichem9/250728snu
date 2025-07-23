# 기존 코드 (문법 오류 있음)
sns.regplot(x=df['Date'].apply(lambda date: date)

# 수정된 코드
from matplotlib.dates import date2num
df['Date_ordinal'] = df['Date'].map(date2num)

sns.regplot(x='Date_ordinal', y=plot_col, data=df, scatter=False, ax=ax, label='추세선')
ax.set_xticks(ax.get_xticks())
ax.set_xticklabels([pd.to_datetime(num).strftime('%Y-%m-%d') for num in ax.get_xticks()], rotation=45)
