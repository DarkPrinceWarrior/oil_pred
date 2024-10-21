import random
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from darts import TimeSeries
from darts.models import TiDEModel
from darts.metrics import mape, rmse
from darts.dataprocessing.transformers import Scaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import torch
from pytorch_lightning.callbacks import EarlyStopping
import re
from tqdm import tqdm
from statsmodels.tsa.stattools import pacf
from statsmodels.graphics.tsaplots import plot_pacf
import random
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from sklearn.model_selection import train_test_split
import re


torch.set_float32_matmul_precision("high")

import pickle 

# Сохранение словаря
def save_dictionary(dictionary, filename):
    with open(filename, 'wb') as file:
        pickle.dump(dictionary, file)

# Загрузка словаря
def load_dictionary(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)
    

# Загрузка
datasets = load_dictionary('datasets_dict_daily_fluid.pkl')
target_feature = "дебит_жидкости"

def find_optimal_chunk_size(data, max_lag=90):
    # Вычисляем частичную автокорреляцию
    pacf_values = pacf(data, nlags=max_lag, method='ywmle')
    
    # Находим значимые лаги (используем 95% доверительный интервал)
    confidence_interval = 1.96 / np.sqrt(len(data))
    significant_lags = np.where(np.abs(pacf_values) > confidence_interval)[0]
    
    # Если нет значимых лагов, возвращаем 1
    if len(significant_lags) == 0:
        return 1
    
    # Находим максимальный значимый лаг
    max_significant_lag = np.max(significant_lags)
    
    return max_significant_lag  # +1, так как лаги начинаются с 0



def spatial_temporal_feature(df, target_col, distance_cols, window_size):
    # Создаем массив расстояний
    distances = df[distance_cols].values
    
    # Создаем веса на основе обратных расстояний
    weights = 1 / (distances + 1)  # +1 чтобы избежать деления на ноль
    
    # Нормализуем веса
    weights = weights / np.sum(weights, axis=1, keepdims=True)
    
    # Создаем скользящее окно для целевой переменной
    rolling_target = df[target_col].rolling(window=window_size, min_periods=1)
    
    # Вычисляем взвешенную сумму
    weighted_sum = np.sum(rolling_target.mean().values[:, np.newaxis] * weights, axis=1)
    
    return weighted_sum

def process_single_well(datasets,target_feature, well_dob_name=None):

    df = datasets[well_dob_name]
    
    # Find optimal chunk size (assuming you have this function defined)
    time_series = df[target_feature].values

    # Calculate difference of target feature
    df[f'{target_feature}_diff'] = df[target_feature].diff().fillna(0)

    # Select relevant columns
    columns_to_use = [col for col in df.columns if not col.startswith('dis_') and col not in ['UWI', 'MD']]
    # columns_to_use = [col for col in columns_to_use if not col.startswith('Приемист_Закачка_воды_KKD_')]
    
    # Создаем новый DataFrame только с нужными колонками
    df_corr = df[columns_to_use].copy()

    # Преобразуем 'Способ_эксплуатации' в категориальную переменную
    df_corr['Способ_эксплуатации'] = pd.Categorical(df_corr['Способ_эксплуатации']).codes

    # Вычисляем корреляционную матрицу
    corr_matrix = df_corr.corr()
    
    correlations = abs(corr_matrix[target_feature])
    correlations = correlations[correlations>0.2]

    # Выбираем нужные столбцы
    columns_to_use = [x for x in columns_to_use if x in list(correlations.index)]
    
    categorical_feat =  ['Способ_эксплуатации','day','month', 'quarter', 'year', 'month_sin', 'month_cos']
    categorical_feat = [x for x in categorical_feat if x in columns_to_use]
    numerical_feat = [x for x in columns_to_use if x not in categorical_feat] 

    pattern_zak_debit = r'Приемист_Закачка_воды_'
    pattern_zak_cum = r'Закачка_с_начала_разр_'
    
    def find_columns(pattern):
        return [col for col in columns_to_use if re.search(pattern, col)]
    
    # Поиск колонок
    zak_columns_debit = find_columns(pattern_zak_debit)
    zak_columns_cum = find_columns(pattern_zak_cum)

    future_covs = ['month', 'quarter', 'year', 'month_sin', 'month_cos','Забойное_давление',
               'Буферное_устьевое_давление','Накоп_время_раб']+zak_columns_debit+zak_columns_cum
    
    future_covs = [x for x in future_covs if x in columns_to_use]
    past_covs = [x for x in columns_to_use if x not in future_covs+[target_feature]]

    # Prepare static covariates
    # Assuming 'df' is your dataframe with static covariates
    static_covs = df[[col for col in df.columns if col.startswith('dis_')]]
    # static_covs_single = static_covs.iloc[[0]].reset_index().drop(columns=["time"])
    static_covs_single = static_covs.iloc[[0]]

    columns = [[y for y in list(static_covs_single.columns) if y[4:]=="_".join(x.split("_")[-2:])] for x in columns_to_use]
    columns = [x[0] for x in columns if len(x)>0]
    static_covs_single = static_covs_single[columns]
    static_covs_single = static_covs_single.reset_index().drop(columns=["Дата"])

    return df, static_covs_single, future_covs, past_covs, target_feature, categorical_feat, numerical_feat

class TargetTransformer:
    def __init__(self, target_column):
        self.target_column = target_column
        self.scaler = MinMaxScaler(feature_range=(-1,1))
    
    def fit(self, series):
        self.scaler.fit(series.values.reshape(-1, 1))
        
    def transform(self, series):
        return pd.Series(self.scaler.transform(series.values.reshape(-1, 1)).flatten(), index=series.index)
    
    def inverse_transform(self, series):
        return pd.Series(self.scaler.inverse_transform(series.values.reshape(-1, 1)).flatten(), index=series.index)

class PastCovariateTransformer:
    def __init__(self, numerical_feat, categorical_feat, past_covs):
        self.numerical_feat = [f for f in numerical_feat if f in past_covs]
        self.categorical_feat = [f for f in categorical_feat if f in past_covs]
        
        transformers = []
        
        if self.numerical_feat:
            numeric_transformer = Pipeline(steps=[
                ('scaler', MinMaxScaler(feature_range=(-1,1)))
            ])
            transformers.append(('num', numeric_transformer, self.numerical_feat))
        
        if self.categorical_feat:
            categorical_transformer = Pipeline(steps=[
                ('onehot', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
            ])
            transformers.append(('cat', categorical_transformer, self.categorical_feat))
        
        self.preprocessor = ColumnTransformer(transformers=transformers)
        
    def fit(self, df):
        self.preprocessor.fit(df)
        
    def transform(self, df):
        transformed = self.preprocessor.transform(df)
        feature_names = []
        if self.numerical_feat:
            feature_names.extend(self.preprocessor.named_transformers_['num'].get_feature_names_out(self.numerical_feat).tolist())
        if self.categorical_feat:
            feature_names.extend(self.preprocessor.named_transformers_['cat'].get_feature_names_out(self.categorical_feat).tolist())
        return pd.DataFrame(transformed, columns=feature_names, index=df.index)

class FutureCovariateTransformer:
    def __init__(self, numerical_feat, categorical_feat, future_covs):
        self.numerical_feat = [f for f in numerical_feat if f in future_covs]
        self.categorical_feat = [f for f in categorical_feat if f in future_covs]
        
        transformers = []
        
        if self.numerical_feat:
            numeric_transformer = Pipeline(steps=[
                ('scaler', MinMaxScaler(feature_range=(-1,1)))
            ])
            transformers.append(('num', numeric_transformer, self.numerical_feat))
        
        if self.categorical_feat:
            categorical_transformer = Pipeline(steps=[
                ('onehot', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
            ])
            transformers.append(('cat', categorical_transformer, self.categorical_feat))
        
        self.preprocessor = ColumnTransformer(transformers=transformers)
        
    def fit_transform(self, df):
        self.preprocessor.fit(df)
        transformed = self.preprocessor.transform(df)
        feature_names = []
        if self.numerical_feat:
            feature_names.extend(self.preprocessor.named_transformers_['num'].get_feature_names_out(self.numerical_feat).tolist())
        if self.categorical_feat:
            feature_names.extend(self.preprocessor.named_transformers_['cat'].get_feature_names_out(self.categorical_feat).tolist())
        return pd.DataFrame(transformed, columns=feature_names, index=df.index)
    
def split_and_transform_data(df, target, past_covs, future_covs, numerical_feat, 
                             categorical_feat, lookback=24, lookahead=12, val_samples=36):
    # for future covs
    df_original = df
    # Remove last lookahead points 
    df = df.iloc[:-lookahead]
    total_samples = len(df)

    # Calculate validation samples (at least one sequence)
    val_samples = val_samples

    # Calculate train samples
    train_samples = total_samples - val_samples

    # Split the data
    train_df = df[:train_samples]
    
    # Initialize transformers
    target_transformer = TargetTransformer(target)
    past_cov_transformer = PastCovariateTransformer(numerical_feat, categorical_feat, past_covs)
    future_cov_transformer = FutureCovariateTransformer(numerical_feat, categorical_feat, future_covs)

    # Fit transformers on training data
    target_transformer.fit(train_df[target])
    past_cov_transformer.fit(train_df[past_covs])

    # Transform data
    transformed_target = target_transformer.transform(df[target])
    transformed_past_covs = past_cov_transformer.transform(df[past_covs])
    transformed_future_covs = future_cov_transformer.fit_transform(df_original[future_covs]) # Fit on all data for future covariates

    # Split transformed data
    train_target = transformed_target[:train_samples]
    train_past_covs = transformed_past_covs[:train_samples]
    # For future covariates, we keep the full range for train and val
    train_future_covs = transformed_future_covs[:train_samples+lookahead]
    
    if val_samples != 0:
        val_target = transformed_target[train_samples:]
        val_past_covs = transformed_past_covs[train_samples:]
        # For future covariates, we keep the full range for train and val
        val_future_covs = transformed_future_covs[train_samples:train_samples+val_samples+lookahead]
        
        return {
            'train': (train_target, train_past_covs, train_future_covs),
            'val': (val_target, val_past_covs, val_future_covs),
            'transformers': {
                'target': target_transformer,
                'past_covs': past_cov_transformer,
                'future_covs': future_cov_transformer
            }
        }
    else:
        return {
            'train': (train_target, train_past_covs, train_future_covs),
            'transformers': {
                'target': target_transformer,
                'past_covs': past_cov_transformer,
                'future_covs': future_cov_transformer
            }
        }
        
def train_model(combined_target, combined_past_covs, combined_future_covs, static_covs_single, lookback, lookahead):
    
    optimizer_kwargs = {
    "lr": 0.001,   # Learning rate
    "betas": (0.9, 0.999),                # Default Adam betas for 1st and 2nd moment
    "eps": 1e-8,                          # Small value to prevent division by zero
    "weight_decay": 1e-4,  # Optional L2 regularization, 0 if not provided
    "amsgrad": False                      # Whether to use the AMSGrad variant of Adam
    }

    lr_scheduler_cls = torch.optim.lr_scheduler.CosineAnnealingLR
    lr_scheduler_kwargs = {
        "T_max":50,
        "eta_min": 1e-6,
    }

    early_stopping = EarlyStopping(
        monitor="train_loss",
        patience=10,
        min_delta=1e-3,
        mode="min",
        check_finite=True,
    )

    pl_trainer_kwargs = {
        "accelerator": "gpu",
        "devices": "auto",
        "callbacks": [early_stopping],
    }

    common_model_args = {
        "input_chunk_length": lookback,
        "output_chunk_length": lookahead,
        "optimizer_kwargs": optimizer_kwargs,
        "pl_trainer_kwargs": pl_trainer_kwargs,
        "lr_scheduler_cls": lr_scheduler_cls,
        "lr_scheduler_kwargs": lr_scheduler_kwargs,
        "likelihood": None,
        "save_checkpoints": False,
        "batch_size": 32,
        "use_layer_norm":True,
        "dropout": 0.001,
        "use_reversible_instance_norm": True,
    }

    # Define the encoders to add past and future covariates
    encoders = {
        "datetime_attribute": {
            "past": ['quarter', 'dayofyear', 'day_of_year',"month", "day"],  # Time attributes for past covariates
            "future": ['quarter', 'dayofyear', 'day_of_year',"month", "day"],  # Time attributes for future covariates
        },
        "cyclic": {
            "past": ['quarter', 'dayofyear', 'day_of_year',"month", "day"],  # Cyclic features for past covariates (helps with seasonal patterns)
            "future": ['quarter', 'dayofyear', 'day_of_year',"month", "day"],  # Cyclic features for future covariates
        },
        
        "position": {
            "past": ["relative"],
            "future": ["relative"]
        }
    }

    # Create the final model
    model = TiDEModel(
        **common_model_args,
        use_static_covariates=True,
        add_encoders=encoders,  # Add the encoders
        model_name="one_shot_model",
        loss_fn=torch.nn.HuberLoss(),
    )

    # Fit the final model on the full training data
    model.fit(
        series=TimeSeries.from_series(combined_target, static_covariates=static_covs_single),
        past_covariates=TimeSeries.from_dataframe(combined_past_covs),
        future_covariates=TimeSeries.from_dataframe(combined_future_covs),
        verbose=False,
        epochs=200,
        dataloader_kwargs={
        "shuffle": False,
        "pin_memory": True
        },
    )
    
    return model


def perform_forecasting(model, combined_target, combined_past_covs, combined_future_covs,
                        static_covs_single, df, target, lookback, lookahead, target_transformer):
    
    
    best_model = model

    series = TimeSeries.from_series(combined_target, static_covariates=static_covs_single)
    past_covariates = TimeSeries.from_dataframe(combined_past_covs)
    future_covariates = TimeSeries.from_dataframe(combined_future_covs)

    # Выполняем historical_forecasts 
    historical_forecasts = best_model.historical_forecasts(
        series=series,
        past_covariates=past_covariates,
        future_covariates=future_covariates,
        forecast_horizon=lookahead,
        train_length = None,
        stride=1,
        retrain=False,
        last_points_only=True,
        overlap_end=False,
        verbose=False
    )

    # Выполняем прогноз
    test_prediction = best_model.predict(
        n=lookahead,
        verbose=False
    )

    # Обратное преобразование прогнозов
    historical_forecasts = TimeSeries.from_series(target_transformer.inverse_transform(historical_forecasts.pd_series()))
    test_forecast = TimeSeries.from_series(target_transformer.inverse_transform(test_prediction.pd_series()))

    # Get the actual values
    actual_series = TimeSeries.from_series(df[target])

    return actual_series, historical_forecasts, test_forecast

def plot_forecasts(actual_series, inversed_hist_forecasts, test_forecast, lookback, lookahead):
    plt.figure(figsize=(20, 10))
    actual_series[lookback+lookahead-1:].plot(label='Actual', color='blue')
    inversed_hist_forecasts.plot(label='Historical Forecasts', color='green')

    # Plot test prediction
    test_prediction_dates = actual_series.time_index[-lookahead:]
    test_forecast.plot(label='Test Forecast', color='red')
    plt.axvline(x=test_prediction_dates[0], color='orange', linestyle='--', label='train/test split')

    plt.legend()
    plt.title('Historical Forecasts, Test Forecast, and Actual Values')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.tight_layout()
    plt.show()
    
def generate_cross_plots(df, train_forecasts, test_forecast):
    # Create DataFrames with forecasted and actual values
    train_data = pd.DataFrame({
        'forecasted_rate': train_forecasts.pd_series(),
        'work_time_days': df.loc[train_forecasts.pd_series().index, "Время_работы_скв_дней"],
        'actual_production': df.loc[train_forecasts.pd_series().index, 'Добыча_жидкости']
    })

    test_data = pd.DataFrame({
        'forecasted_rate': test_forecast.pd_series(),
        'work_time_days': df.loc[test_forecast.pd_series().index, "Время_работы_скв_дней"],
        'actual_production': df.loc[test_forecast.pd_series().index, 'Добыча_жидкости']
    })

    # Calculate forecasted production
    train_data['forecasted_production'] = train_data['forecasted_rate'] * train_data['work_time_days']
    test_data['forecasted_production'] = test_data['forecasted_rate'] * test_data['work_time_days']

    # Create subplots
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Training Set", "Test Set"))

    # Function to add points and y=x line to subplot
    def add_subplot(data, row, col):
        max_value = max(data['actual_production'].max(), data['forecasted_production'].max())
        
        fig.add_trace(
            go.Scatter(
                x=data['actual_production'],
                y=data['forecasted_production'],
                mode='markers',
                marker=dict(size=8, color='blue', opacity=0.6),
                name='Forecast vs Actual',
                hovertemplate='<b>Date:</b> %{customdata}<br>' +
                              '<b>Actual production:</b> %{x:.2f}<br>' +
                              '<b>Forecasted production:</b> %{y:.2f}<extra></extra>',
                customdata=data.index.strftime('%Y-%m-%d')
            ),
            row=row, col=col
        )
        
        fig.add_trace(
            go.Scatter(
                x=[0, max_value],
                y=[0, max_value],
                mode='lines',
                line=dict(color='red', dash='dash'),
                name='y=x (perfect fit)',
                showlegend=False
            ),
            row=row, col=col
        )

    # Add data to subplots
    add_subplot(train_data, 1, 1)
    add_subplot(test_data, 1, 2)

    # Configure layout
    fig.update_layout(
        title='Cross-plot: Forecasted vs Actual Fluid Production',
        height=600,
        width=1200,
        showlegend=True
    )

    fig.update_xaxes(title_text="Actual Fluid Production", row=1, col=1)
    fig.update_xaxes(title_text="Actual Fluid Production", row=1, col=2)
    fig.update_yaxes(title_text="Forecasted Fluid Production", row=1, col=1)
    fig.update_yaxes(title_text="Forecasted Fluid Production", row=1, col=2)

    # Display the plot
    fig.show()

    # Calculate statistics
    def calculate_stats(data):
        mape = ((data['actual_production'] - data['forecasted_production']).abs() / data['actual_production']).mean() * 100
        correlation = data['actual_production'].corr(data['forecasted_production'])
        return mape, correlation

    train_mape, train_corr = calculate_stats(train_data)
    test_mape, test_corr = calculate_stats(test_data)

    print("Statistics for the training set:")
    print(f"Mean Absolute Percentage Error (MAPE): {train_mape:.2f}%")
    print(f"Correlation between actual and forecasted production: {train_corr:.2f}")

    print("\nStatistics for the test set:")
    print(f"Mean Absolute Percentage Error (MAPE): {test_mape:.2f}%")
    print(f"Correlation between actual and forecasted production: {test_corr:.2f}")
    
    
def process_well(df, static_covs_single, future_covs, past_covs, target_feature,categorical_feat, numerical_feat):
    
    lookback = 360
    lookahead = 180
    val_samples = 0  # We're not using a validation set in this example

    split_data = split_and_transform_data(
        df, 
        target_feature, 
        past_covs, 
        future_covs, 
        numerical_feat, 
        categorical_feat, 
        lookback=lookback, 
        lookahead=lookahead,
        val_samples=val_samples
    )

    combined_target, combined_past_covs, combined_future_covs = split_data['train']
    target_transformer = split_data['transformers']['target']

    model = train_model(combined_target, combined_past_covs, combined_future_covs, static_covs_single, lookback, lookahead)

    actual_series, historical_forecasts, test_forecast = perform_forecasting(
        model, combined_target, combined_past_covs, combined_future_covs, static_covs_single,
        df, target_feature, lookback, lookahead, target_transformer
    )

    # Combine historical forecasts and test forecast
    all_forecasts = historical_forecasts.append(test_forecast)

    if target_feature=="дебит_жидкости":
        # Calculate forecasted production
        forecasted_production = all_forecasts.pd_series().resample('M').sum()
        # Get actual production directly from the dataframe
        actual_production = df['дебит_жидкости'].loc[all_forecasts.pd_series().index].resample('M').sum()
        
        return actual_production, forecasted_production,actual_series,historical_forecasts,test_forecast
    
    # Переделать
    else:
        forecasted_water_cut = all_forecasts.pd_series()
        # Для forecasted_water_cut
        forecasted_water_cut = forecasted_water_cut.clip(upper=100) # Поставить 100
        
        return df.loc[forecasted_water_cut.index, "Обводненность"].resample('M').sum(),forecasted_water_cut.resample('M').sum(),actual_series,"",""

# Main execution
def main(datasets,target):
    results = {}

    for well_name in tqdm(datasets.keys(), desc="Processing wells"):
        
        df, static_covs_single, future_covs, past_covs, target_feature, categorical_feat, numerical_feat = process_single_well(datasets,target,well_name)
        
        # if len(df)<3000:
        #     continue
    
        actual, forecasted,actual_series,historical_forecasts,test_forecast = process_well(df, static_covs_single, future_covs, past_covs, 
                                                                                           target_feature, categorical_feat, numerical_feat)
        
        if target_feature=="дебит_жидкости":
            # Store results
            results[well_name] = {
                'actual_production': actual,
                'forecasted_production': forecasted
            }
        else:
            # Store results
            results[well_name] = {
                'actual_water_cut': actual,
                'forecasted_water_cut': forecasted
            }
            
        

    # After processing all wells, you can perform additional analysis or visualization on the results dictionary
    print("\nAll wells processed. Results stored in 'results' dictionary.")

    return results
    # You can add more analysis here, such as aggregating results across all wells

if __name__ == "__main__":
    # Assuming you have the datasets dictionary defined
    results = main(datasets,target_feature)
    save_dictionary(results, 'results_fluid_prod.pkl')
    
    