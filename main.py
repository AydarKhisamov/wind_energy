import os
import re

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor

# фиксация конфигов
DATA_PATH = 'data'
MODELS_PATH = 'models'
SUBMISSION_PATH = 'submissions'

dt_col = 'METEOFORECASTHOUR_OPENM_Datetime'
target_col = 'Выработка. Результирующий расчет'

# чтение файла с размеченными данными
df = pd.read_csv(f'{DATA_PATH}/train_dataset.csv')

# замещение пропущенных значений в столбцах wind_direction_180m, wind_speed_180m
# с помощью алгоритма линейной регрессии, использующей аналогичные признаки на
# более низкой высоте - 10м, 80м, 120м
scaler = StandardScaler()
model = LinearRegression()

for metric in ['wind_speed', 'wind_direction']:
    no_nan_idxs = df[df[f'{metric}_180m'].notna()].index
    nan_idxs = df[df[f'{metric}_180m'].isna()].index

    # нормализация признаков
    x_train = scaler.fit_transform(
        df.loc[
            no_nan_idxs, [f'{metric}_{h}' for h in ['10m', '80m', '120m']]
        ],
    )
    x_test = scaler.transform(
        df.loc[nan_idxs, [f'{metric}_{h}' for h in ['10m', '80m', '120m']]],
    )

    # нормализация целевого признака
    y_train = scaler.fit_transform(
        df.loc[no_nan_idxs, f'{metric}_180m'].values.reshape(-1, 1)
    )

    # обучение алгоритма линейной регрессии
    model.fit(x_train, y_train)

    # предсказание пропущенных значений
    y_predicted = scaler.inverse_transform(model.predict(x_test))
    df.loc[nan_idxs, f'{metric}_180m'] = y_predicted

# конвертация дат в тип datetime и сортировка по времени по возрастанию
df[dt_col] = pd.to_datetime(df[dt_col])
df = df.sort_values(dt_col)

# создание обучающей выборки, в которой не будет пропущенных datetime:
train_df = pd.DataFrame(
    {dt_col: pd.date_range(df[dt_col].min(), df[dt_col].max(), freq='h')}
)

# линейная интерполяция пропущенных признаков по соседним (до и после)
train_df = train_df.merge(df, on=dt_col, how='left')
train_df['month'] = train_df[dt_col].dt.month
train_df['hour_of_day'] = train_df[dt_col].dt.hour
train_df = train_df.interpolate(limit_direction='both')

# создание специального датафрейма, который:
#   1) будет использоваться для обогащения данных признаками;
#   2) будет дополняться новыми данными (recursive prediction);
stats_df = train_df.copy()


def prepare_features(raw_data: pd.DataFrame):
    """
    Создаёт признаки для каждого datetime.

    Args:
        raw_data: датафрейм с почасовыми данными по электрогенерации;

    Returns:
        feature_store: словарь с датафреймами с признак(-ом/-ами) и datetime
            в качестве ключа для присоединения.
    """
    features_store = {}

    # средний объём генерации за последние {window} часов
    for window in ['1h', '4h', '8h', '24h']:
        n_periods = int(re.search(r'^\d+', window, flags=re.M)[0])
        temp_df = (
            raw_data
            .set_index(dt_col)
            .rolling(window=window, min_periods=n_periods)
            .agg({target_col: 'mean'})
            .dropna()
            .rename(columns={target_col: f'last_{window}_target_mean'})
        )

        if re.search(r'[a-z]+$', window, flags=re.M)[0] == 'h':
            offset_args = {'hours': 1}

        temp_df.index += pd.DateOffset(**offset_args)
        features_store[f'last_{window}_target_mean'] = temp_df

    # 1) кол-во часов подряд, когда КИУМ < {capacity}
    # 2) кол-во прошедших часов с момента, когда было: КИУМ < {capacity}
    for capacity in [0.01, 0.05, 0.1]:
        temp_df = raw_data[
            [dt_col, target_col, 'Кол-во_ВЭУ_в_ремонте']
        ].copy()

        temp_df['max_production'] = (
            90.09 - temp_df['Кол-во_ВЭУ_в_ремонте'] * 3.465
        )

        temp_df['power_capacity'] = (
            temp_df[target_col] / temp_df['max_production']
        )

        mask = temp_df['power_capacity'] < capacity
        temp_df[f'capacity_lt{capacity}_time'] = (
            mask
            .groupby((~mask).cumsum())
            .cumsum()
            .astype(int)
        )

        mask = temp_df['power_capacity'] >= capacity
        temp_df[f'time_from_last_capacity_lt{capacity}'] = (
            mask
            .groupby((~mask).cumsum())
            .cumsum()
            .astype(int)
        )

        temp_df[dt_col] += pd.DateOffset(hours=1)

        for col in [
            f'capacity_lt{capacity}_time',
            f'time_from_last_capacity_lt{capacity}',
        ]:
            features_store[col] = temp_df[[dt_col, col]]

    # кол-во часов подряд, когда выработка энергии повышается
    temp_df = raw_data[[dt_col, target_col]].copy()
    mask = temp_df[target_col] >= temp_df[target_col].shift()
    temp_df['target_increasing_time'] = (
        mask
        .groupby((~mask).cumsum())
        .cumsum()
        .astype(int)
    )

    # кол-во часов подряд, когда выработка энергии снижается
    mask = temp_df[target_col] < temp_df[target_col].shift()
    temp_df['target_decreasing_time'] = (
        mask
        .groupby((~mask).cumsum())
        .cumsum()
        .astype(int)
    )
    
    # отношение объёма выработки 1ч назад к объёму выработки 2ч назад
    temp_df['last_hour_momentum'] = (
        temp_df[target_col] / temp_df[target_col].shift()
    )

    temp_df[dt_col] += pd.DateOffset(hours=1)
    features_store['target_momentum'] = temp_df.drop(columns=target_col)

    # накопленный КИУМ на момент времени суток;
    # для прогноза на 00:00 равен 0;
    temp_df = raw_data[[dt_col, target_col, 'Кол-во_ВЭУ_в_ремонте']].copy()
    temp_df['period'] = temp_df[dt_col].dt.to_period('D')
    temp_df['max_production'] = (
        90.09 - temp_df['Кол-во_ВЭУ_в_ремонте'] * 3.465
    )

    temp_cumsums = (
        temp_df
        .groupby('period')[[target_col, 'max_production']]
        .cumsum()
    )

    temp_cumsums['inday_target_to_max_ratio'] = (
        temp_cumsums[target_col] / temp_cumsums['max_production']
    )

    temp_df = pd.concat(
        [temp_df, temp_cumsums['inday_target_to_max_ratio']], axis=1,
    )

    temp_df[dt_col] += pd.DateOffset(hours=1)
    temp_df.loc[
        temp_df[dt_col].dt.hour == 0, 'inday_target_to_max_ratio'
    ] = 0

    features_store['inday_target_to_max_ratio'] = (
        temp_df[[dt_col, 'inday_target_to_max_ratio']]
    )

    return features_store


# подготовка признаков и их добавление в обучающую выборку
features = prepare_features(stats_df)
for feature in features.values():
    train_df = train_df.merge(feature, on=dt_col)

# сохранение предобработанной обучающей выборки по условию задачи
train_df = train_df.drop(columns=dt_col)
train_df.to_csv(f'{DATA_PATH}/processed_train_dataset.csv', index=False)

# инициализация модели градиентного бустинга для прогноза генерации энергии
model = CatBoostRegressor(
    iterations=5000,
    random_strength=3,
    l2_leaf_reg=7.0,
    use_best_model=True,
    logging_level='Silent',
    random_state=42,
    early_stopping_rounds=200,
)

# разбиение выборки на обучающую и валидационную со стратификацией по месяцам
train_idxs, val_idxs = train_test_split(
    train_df.index,
    test_size=0.2,
    random_state=42,
    stratify=train_df['month'],
)

x_train = train_df.loc[train_idxs].drop(columns=target_col)
y_train = train_df.loc[train_idxs, target_col]
x_val = train_df.loc[val_idxs].drop(columns=target_col)
y_val = train_df.loc[val_idxs, target_col]

# обучение модели и её сохранение по условию задачи
model.fit(X=x_train, y=y_train, eval_set=(x_val, y_val))
if not os.path.isdir(MODELS_PATH):
    os.mkdir(MODELS_PATH)

model.save_model(f'{MODELS_PATH}/model.cbm')

# подготовка тестовой выборки
valid_features = pd.read_csv(f'{DATA_PATH}/valid_features.csv')
valid_features[dt_col] = pd.to_datetime(valid_features[dt_col])
valid_features = valid_features.sort_values(dt_col)

test_df = pd.DataFrame(
    {dt_col: pd.date_range(
        valid_features[dt_col].min(),
        valid_features[dt_col].max(),
        freq='h',
    )}
)

test_df = test_df.merge(valid_features, on=dt_col, how='left')
test_df['month'] = test_df[dt_col].dt.month
test_df['hour_of_day'] = test_df[dt_col].dt.hour
test_df = test_df.interpolate(limit_direction='both')

# recursive prediction:
predictions = []
while len(test_df) > 0:
    temp_df = test_df.copy()
    # добавление в тестовую выборку признаков
    for f in features.values():
        temp_df = temp_df.merge(f, on=dt_col)

    # прогноз
    temp_idxs = temp_df.index
    temp_df[target_col] = (
        model.predict(temp_df.drop(columns=dt_col))
    )
    predictions.append(temp_df[[dt_col, target_col]])

    # добавление прогноза в датафрейм со статистикой
    stats_df = pd.concat([stats_df, temp_df], ignore_index=True)

    # удаление из тестовой выборки записи, для которой уже есть прогноз
    test_df = test_df.drop(index=test_df.iloc[temp_idxs].index)

    # создание признаков для новых дат
    features = prepare_features(stats_df)

submission = pd.concat(predictions, ignore_index=True)

# сохранение прогнозов только тех дат, которые есть в тестовой выборке
submission = submission.merge(valid_features[[dt_col]], on=dt_col)
if not os.path.isdir(SUBMISSION_PATH):
    os.mkdir(SUBMISSION_PATH)

submission = submission.sort_values(dt_col, ascending=False)
submission[[target_col]].to_csv(f'{SUBMISSION_PATH}/submission_1q.csv', index=False)

# чтение файла с актуальными данными
last_day_df = pd.read_csv(f'{DATA_PATH}/3888f9f2-9bda-4b2c-94af-5562668bce86_test_dataset.csv')
last_day_df[dt_col] = pd.to_datetime(last_day_df[dt_col])
last_day_df = last_day_df.sort_values(dt_col)

# создание тестовой выборки 
test_df = pd.DataFrame(
    {dt_col: pd.date_range(
        pd.Timestamp('2026-05-18 00:00:00'),
        pd.Timestamp('2026-05-18 23:00:00'),
        freq='h',
    )}
)

test_df = pd.merge_asof(
    test_df, last_day_df, on=dt_col, tolerance=pd.Timedelta('1s'),
).drop(columns=target_col)

# добавление в датафрейм со статистикой новых доступных данных
stats_df = pd.concat([
    stats_df,
    last_day_df[last_day_df[dt_col] <= pd.Timestamp('2026-05-17 23:00:00')],
])

# генерация признаков для новых дат
features = prepare_features(stats_df)

# создание почасового прогноза на 18.05.2026
predictions = []
while len(test_df) > 0:
    temp_df = test_df.copy()
    for f in features.values():
        temp_df = temp_df.merge(f, on=dt_col)

    temp_idxs = temp_df.index
    temp_df[target_col] = (
        model.predict(temp_df.drop(columns=dt_col))
    )
    predictions.append(temp_df[[dt_col, target_col]])
    stats_df = pd.concat([stats_df, temp_df], ignore_index=True)
    test_df = test_df.drop(index=test_df.iloc[temp_idxs].index)
    features = prepare_features(stats_df)

submission = pd.concat(predictions, ignore_index=True)
submission = submission.sort_values(dt_col, ascending=False)
submission[[target_col]].to_csv(
    f'{SUBMISSION_PATH}/submission_18-05-2026.csv', index=False,
)
