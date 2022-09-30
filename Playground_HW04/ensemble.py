import pandas as pd

csv_list = []
ensemble_times = 5
for i in range(ensemble_times):
    csv_list.append(pd.read_csv(f'output_{i}.csv'))
for i in range(1, ensemble_times):
    csv_list[0][f'Category{i}'] = csv_list[i]['Category']

csv_list[0]['Category'] = csv_list[0].drop(['Id'], axis=1).mode(1)[0]
csv_list[0][['Id', 'Category']].to_csv('sub.csv', index=False)
