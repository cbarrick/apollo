import numpy as np
import pandas as pd

from xgboost import XGBRegressor
from sklearn.model_selection import KFold, GridSearchCV

from apollo.datasets import SolarDataset


def main():
    for lag in [1, 2, 4, 8]:
        dataset = SolarDataset(
            start = '2017-01-01 00:00',
            stop = '2017-12-31 18:00',
            forecast = 0,
            lag = lag,
        )

        x, y = dataset.tabular()
        x = np.asarray(x)
        y = np.asarray(y)

        model = GridSearchCV(
            estimator = XGBRegressor(),
            param_grid = {
                'n_estimators': [32, 64, 128, 256, 512],
                'max_depth': [8, 16, 32, 64, 128],
            },
            cv = KFold(n_splits=3, shuffle=True),
            scoring = 'neg_mean_absolute_error',
            return_train_score = False,
            n_jobs = -1,
        ).fit(x, y)

        results = pd.DataFrame(model.cv_results_)
        results = results.set_index('rank_test_score')
        results = results.sort_index()
        results = results[['mean_test_score', 'std_test_score', 'param_max_depth', 'param_n_estimators']]
        print(f'lag: {lag}')
        print(results)
        print()


if __name__ == '__main__':
    main()
