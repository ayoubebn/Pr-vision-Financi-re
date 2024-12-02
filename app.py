import dash
from dash import dcc, html, Input, Output
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Initialiser l'application Dash
app = dash.Dash(__name__)

# Définir l'interface utilisateur
app.layout = html.Div([
    html.H1("Analyse de Séries Temporelles - Yahoo Finance"),
    html.Div([
        html.Label("Entrer le symbole de l'organisation"),
        dcc.Input(id='sym', type='text', value='HDFCBANK.NS'),
        html.Label("Sélectionnez la plage de dates"),
        dcc.DatePickerRange(
            id='dates',
            start_date='2023-01-01',
            end_date='2024-01-01'
        ),
        html.Label("Pourcentage d'entraînement (%)"),
        dcc.Slider(id='split', min=80, max=95, step=1, value=90, marks={i: f'{i}%' for i in range(80, 96, 5)}),
        html.Label("Horizon de prédiction"),
        dcc.Slider(id='horizon', min=5, max=10, step=1, value=5),
        html.Label("Paramètres ARIMA/SARIMA"),
        html.Label("Autoregression (p)"),
        dcc.Slider(id='parp', min=0, max=5, step=1, value=2),
        html.Label("Intégration (d)"),
        dcc.Slider(id='pard', min=0, max=2, step=1, value=1),
        html.Label("Moyenne mobile (q)"),
        dcc.Slider(id='parq', min=0, max=5, step=1, value=2),
        html.Label("Paramètres saisonniers pour SARIMA"),
        dcc.Slider(id='sp', min=0, max=3, step=1, value=1, marks={i: f'S{i}' for i in range(4)}),
        dcc.Slider(id='sq', min=0, max=3, step=1, value=1, marks={i: f'S{i}' for i in range(4)}),
        dcc.Input(id='seasonal_period', type='number', value=12, placeholder='Période saisonnière (ex : 12)'),
        html.Label("Modèle à utiliser"),
        dcc.Dropdown(
            id='model',
            options=[
                {'label': 'ARIMA', 'value': 'ARIMA'},
                {'label': 'ARIMAX', 'value': 'ARIMAX'},
                {'label': 'SARIMA', 'value': 'SARIMA'}
            ],
            value='ARIMA'
        ),
        html.Label("Variable explicative (pour ARIMAX)"),
        dcc.Input(id='exog', type='text', placeholder='Symbole de la variable explicative (ex: SPY)')
    ], style={'width': '25%', 'float': 'left', 'padding': '10px'}),
    html.Div([
        dcc.Graph(id='ts_plot'),
        html.Div(id='metrics')
    ], style={'width': '70%', 'float': 'right'})
])

def get_data(symbol, start_date, end_date):
    """
    Télécharge les données boursières et les prépare pour l'analyse
    """
    try:
        data = yf.download(symbol, start=start_date, end=end_date)
        data = data['Adj Close'].resample('D').last().ffill()
        if data.empty:
            raise ValueError("Aucune donnée disponible pour les dates spécifiées.")
        return data
    except Exception as e:
        print(f"Erreur lors du téléchargement des données : {e}")
        return None

def prepare_data(data, split_ratio):
    """
    Prépare les données pour l'entraînement et le test
    """
    # Supprimer les valeurs NaN
    data = data.dropna()
    
    # Vérifier le nombre suffisant de points de données
    if len(data) < 30:
        raise ValueError("Nombre insuffisant de points de données pour l'analyse")
    
    # Diviser les données
    n = len(data)
    split_point = int(n * split_ratio / 100)
    train_data = data[:split_point]
    test_data = data[split_point:]
    
    return train_data, test_data

@app.callback(
    [Output('ts_plot', 'figure'),
     Output('metrics', 'children')],
    [Input('sym', 'value'),
     Input('dates', 'start_date'),
     Input('dates', 'end_date'),
     Input('split', 'value'),
     Input('horizon', 'value'),
     Input('parp', 'value'),
     Input('pard', 'value'),
     Input('parq', 'value'),
     Input('sp', 'value'),
     Input('sq', 'value'),
     Input('seasonal_period', 'value'),
     Input('model', 'value'),
     Input('exog', 'value')]
)
def update_plots(sym, start_date, end_date, split, horizon, parp, pard, parq, sp, sq, seasonal_period, model, exog):
    """
    Mettre à jour les graphiques et les métriques en fonction des paramètres sélectionnés
    """
    try:
        # Charger les données
        data = get_data(sym, start_date, end_date)
        if data is None or data.empty:
            return go.Figure(), "Impossible de charger les données ou aucune donnée disponible."

        train_data, test_data = prepare_data(data, split)

        # Charger les données explicatives pour ARIMAX
        exog_data = None
        if model == 'ARIMAX' and exog:
            try:
                exog_data = get_data(exog, start_date, end_date)
                if exog_data is None or exog_data.empty:
                    return go.Figure(), "Erreur avec la variable explicative : aucune donnée disponible."
                exog_train, exog_test = prepare_data(exog_data, split)
            except Exception as e:
                return go.Figure(), f"Erreur avec la variable explicative: {str(e)}"
        else:
            exog_train = exog_test = None

        # Sélection et entraînement du modèle
        try:
            if model == 'ARIMA':
                fitted_model = SARIMAX(train_data, order=(parp, pard, parq)).fit(disp=False)
                predictions = fitted_model.get_forecast(steps=len(test_data)).predicted_mean
            elif model == 'ARIMAX':
                fitted_model = SARIMAX(train_data, exog=exog_train, order=(parp, pard, parq)).fit(disp=False)
                predictions = fitted_model.get_forecast(steps=len(test_data), exog=exog_test).predicted_mean
            elif model == 'SARIMA':
                fitted_model = SARIMAX(train_data, order=(parp, pard, parq),
                                       seasonal_order=(sp, pard, sq, seasonal_period)).fit(disp=False)
                predictions = fitted_model.get_forecast(steps=len(test_data)).predicted_mean
            else:
                return go.Figure(), "Modèle inconnu"
        except Exception as model_err:
            return go.Figure(), f"Erreur de modélisation: {str(model_err)}"

        # Évaluation des performances
        mae = mean_absolute_error(test_data, predictions)
        rmse = np.sqrt(mean_squared_error(test_data, predictions))

        # Création du graphique
        ts_fig = go.Figure()
        ts_fig.add_trace(go.Scatter(x=data.index, y=data, mode='lines', name='Données Réelles'))
        ts_fig.add_trace(go.Scatter(x=test_data.index, y=predictions, mode='lines', name=f'{model} Prédictions'))
        ts_fig.update_layout(title=f'Série Temporelle - {sym}', xaxis_title='Temps', yaxis_title='Prix Ajusté')

        # Métriques
        metrics = html.Div([
            html.H4("Évaluation des Modèles"),
            html.P(f"MAE (Erreur Absolue Moyenne) : {mae:.2f}"),
            html.P(f"RMSE (Racine Carrée de l'Erreur Quadratique Moyenne) : {rmse:.2f}")
        ])

        return ts_fig, metrics

    except Exception as e:
        return go.Figure(), f"Erreur générale: {str(e)}"

# Lancer l'application
if __name__ == '__main__':
    app.run_server(debug=True)