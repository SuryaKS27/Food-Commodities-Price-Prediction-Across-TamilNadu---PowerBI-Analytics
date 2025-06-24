import pandas as pd
import plotly.express as px
import dash
from dash import dcc, html, Input, Output
from prophet import Prophet  # Using Prophet for time series forecasting
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import mysql.connector  # To connect and interact with MySQL
import numpy as np

# Load and preprocess the dataset
df = pd.read_csv('final_data.csv')
df['date'] = pd.to_datetime(df['date'])  # Convert date to datetime format

# Remove outliers based on modal_price (e.g., beyond 3 standard deviations)
def remove_outliers(data, column):
    mean = data[column].mean()
    std_dev = data[column].std()
    return data[(data[column] >= mean - 3 * std_dev) & (data[column] <= mean + 3 * std_dev)]

df = remove_outliers(df, 'modal_price')

# Create additional time-based features
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['day_of_week'] = df['date'].dt.dayofweek
df['year'] = df['date'].dt.year

# Create the Dash app
app = dash.Dash(__name__)

# Connect to MySQL database
conn = mysql.connector.connect(
    host='localhost',  # Update with your MySQL host
    user='root',       # Update with your MySQL username
    password='',       # Update with your MySQL password
    database='commodity_data'  # Database name
)
cursor = conn.cursor()

# Create tables to store forecast results for Prophet and Random Forest
create_table_prophet = """
CREATE TABLE IF NOT EXISTS forecast_results_prophet (
    id INT AUTO_INCREMENT PRIMARY KEY,
    forecast_date DATE,
    forecast_price FLOAT,
    state_name VARCHAR(255),
    district_name VARCHAR(255),
    market_name VARCHAR(255),
    commodity_name VARCHAR(255)
);
"""

create_table_rf = """
CREATE TABLE IF NOT EXISTS forecast_results_rf (
    id INT AUTO_INCREMENT PRIMARY KEY,
    forecast_date DATE,
    forecast_price FLOAT,
    state_name VARCHAR(255),
    district_name VARCHAR(255),
    market_name VARCHAR(255),
    commodity_name VARCHAR(255)
);
"""

cursor.execute(create_table_prophet)
cursor.execute(create_table_rf)
conn.commit()

# Layout for the dashboard
app.layout = html.Div([
    html.H1('Commodity Price Forecast Dashboard', style={'textAlign': 'center', 'padding': '20px'}),
    
    html.Div([
        html.Div([
            html.Label('Select State:'),
            dcc.Dropdown(
                id='state-dropdown', 
                options=[{'label': state, 'value': state} for state in df['state_name'].unique()],
                value=df['state_name'].unique()[0]
            ),
        ], style={'width': '24%', 'display': 'inline-block', 'padding': '10px'}),

        html.Div([
            html.Label('Select District:'),
            dcc.Dropdown(id='district-dropdown'),
        ], style={'width': '24%', 'display': 'inline-block', 'padding': '10px'}),

        html.Div([
            html.Label('Select Market:'),
            dcc.Dropdown(id='market-dropdown'),
        ], style={'width': '24%', 'display': 'inline-block', 'padding': '10px'}),

        html.Div([
            html.Label('Select Commodity:'),
            dcc.Dropdown(id='commodity-dropdown'),
        ], style={'width': '24%', 'display': 'inline-block', 'padding': '10px'}),
    ]),

    html.Div([
        html.Div([
            html.Label('Enter Forecast Period (Days):'),
            dcc.Input(
                id='forecast-period-input', 
                type='number', 
                value=30, 
                min=1, 
                step=1, 
                placeholder='Enter the number of days to forecast'
            ),
        ], style={'width': '48%', 'display': 'inline-block', 'padding': '10px'}),

        html.Div([
            html.Label('Select Model:'),
            dcc.RadioItems(
                id='model-selection',
                options=[
                    {'label': 'Prophet', 'value': 'prophet'},
                    {'label': 'Random Forest', 'value': 'random_forest'}
                ],
                value='prophet',
                inline=True
            ),
        ], style={'width': '48%', 'display': 'inline-block', 'padding': '10px'}),
    ]),

    dcc.Graph(id='historical-price-graph', style={'margin-top': '30px'}),

    dcc.Graph(id='price-forecast-graph', style={'margin-top': '30px'}),

    html.Div(id='forecast-metrics', style={'padding': '20px'}),

    dcc.Graph(id='heatmap', style={'margin-top': '30px'})
])

# Update district dropdown based on selected state
@app.callback(
    Output('district-dropdown', 'options'),
    Input('state-dropdown', 'value')
)
def update_district_dropdown(selected_state):
    districts = df[df['state_name'] == selected_state]['district_name'].unique()
    return [{'label': district, 'value': district} for district in districts]

# Update market dropdown based on selected district
@app.callback(
    Output('market-dropdown', 'options'),
    Input('district-dropdown', 'value')
)
def update_market_dropdown(selected_district):
    markets = df[df['district_name'] == selected_district]['market_name'].unique()
    return [{'label': market, 'value': market} for market in markets]

# Update commodity dropdown based on selected market
@app.callback(
    Output('commodity-dropdown', 'options'),
    Input('market-dropdown', 'value')
)
def update_commodity_dropdown(selected_market):
    commodities = df[df['market_name'] == selected_market]['commodity_name'].unique()
    return [{'label': commodity, 'value': commodity} for commodity in commodities]

# Plot historical price trends
@app.callback(
    Output('historical-price-graph', 'figure'),
    [Input('state-dropdown', 'value'), 
     Input('district-dropdown', 'value'), 
     Input('market-dropdown', 'value'), 
     Input('commodity-dropdown', 'value')]
)
def plot_historical_prices(selected_state, selected_district, selected_market, selected_commodity):
    if None in [selected_state, selected_district, selected_market, selected_commodity]:
        return px.line(title='Select all filters to view historical prices')

    # Filter data
    filtered_df = df[(df['state_name'] == selected_state) &
                     (df['district_name'] == selected_district) &
                     (df['market_name'] == selected_market) &
                     (df['commodity_name'] == selected_commodity)]

    fig = px.line(
        filtered_df, x='date', y='modal_price',
        title=f'Historical Prices for {selected_commodity} in {selected_market}, {selected_district}, {selected_state}',
        labels={'date': 'Date', 'modal_price': 'Modal Price'}
    )
    return fig

from sklearn.metrics import mean_squared_error, r2_score
import math

@app.callback(
    [Output('price-forecast-graph', 'figure'), Output('forecast-metrics', 'children')],
    [Input('state-dropdown', 'value'), 
     Input('district-dropdown', 'value'), 
     Input('market-dropdown', 'value'), 
     Input('commodity-dropdown', 'value'), 
     Input('forecast-period-input', 'value'),
     Input('model-selection', 'value')]
)
def update_graph(selected_state, selected_district, selected_market, selected_commodity, forecast_period, selected_model):
    if None in [selected_state, selected_district, selected_market, selected_commodity, forecast_period]:
        return px.line(title='Select all filters and enter a forecast period to view forecast'), ''

    # Filter the data
    filtered_df = df[(df['state_name'] == selected_state) & 
                     (df['district_name'] == selected_district) & 
                     (df['market_name'] == selected_market) & 
                     (df['commodity_name'] == selected_commodity)]

    if selected_model == 'prophet':
        # Prepare the data for Prophet
        prophet_df = filtered_df[['date', 'modal_price']].rename(columns={'date': 'ds', 'modal_price': 'y'})

        if len(prophet_df) < 30:  # Check if there is enough data to train the model
            return px.line(title='Not enough data to create a forecast'), ''

        # Fit the Prophet model
        model = Prophet(seasonality_mode='multiplicative', yearly_seasonality=True)
        model.add_seasonality(name='monthly', period=30.5, fourier_order=3)
        model.fit(prophet_df)

        # Make future predictions for the user-defined forecast period
        future = model.make_future_dataframe(periods=forecast_period, include_history=False)
        forecast = model.predict(future)

        # Store forecast data in the Prophet MySQL table
        forecast_data = forecast[['ds', 'yhat']].rename(columns={'ds': 'forecast_date', 'yhat': 'forecast_price'})
        
        # for index, row in forecast_data.iterrows():
        #     cursor.execute("""
        #         INSERT INTO forecast_results_prophet (forecast_date, forecast_price, state_name, district_name, market_name, commodity_name) 
        #         VALUES (%s, %s, %s, %s, %s, %s)
        #     """, (row['forecast_date'], row['forecast_price'], selected_state, selected_district, selected_market, selected_commodity))

        # conn.commit()
        for index, row in forecast_data.iterrows():
            forecast_date = row['forecast_date'].date()  # Convert Timestamp -> date
            cursor.execute("""
                INSERT INTO forecast_results_prophet (forecast_date, forecast_price, state_name, district_name, market_name, commodity_name) 
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (forecast_date, row['forecast_price'], selected_state, selected_district, selected_market, selected_commodity))
       

        # Plot forecast
        fig = px.line(
            forecast, x='ds', y='yhat', 
            title=f'Price Forecast for {selected_commodity} in {selected_market}, {selected_district}, {selected_state}',
            labels={'ds': 'Date', 'yhat': 'Forecast Price'}
        )

        return fig, ''

    elif selected_model == 'random_forest':
        # Prepare the data for Random Forest
        rf_df = filtered_df.copy()
        rf_df['days_from_start'] = (rf_df['date'] - rf_df['date'].min()).dt.days

        X = rf_df[['days_from_start', 'month', 'day', 'day_of_week', 'year']]
        y = rf_df['modal_price']

        if len(X) < 30:  # Ensure enough data for training
            return px.line(title='Not enough data to create a forecast'), ''

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train the Random Forest Regressor
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)

        # Generate predictions for the forecast period
        last_date = rf_df['date'].max()
        future_days = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_period)
        future_features = pd.DataFrame({
            'days_from_start': (future_days - rf_df['date'].min()).days,
            'month': future_days.month,
            'day': future_days.day,
            'day_of_week': future_days.dayofweek,
            'year': future_days.year
        })

        forecast_prices = rf_model.predict(future_features)
        forecast_df = pd.DataFrame({'forecast_date': future_days, 'forecast_price': forecast_prices})

        # # Store forecast data in the Random Forest MySQL table
        # for index, row in forecast_df.iterrows():
        #     cursor.execute("""
        #         INSERT INTO forecast_results_rf (forecast_date, forecast_price, state_name, district_name, market_name, commodity_name) 
        #         VALUES (%s, %s, %s, %s, %s, %s)
        #     """, (row['forecast_date'], row['forecast_price'], selected_state, selected_district, selected_market, selected_commodity))
        for index, row in forecast_df.iterrows():
            forecast_date = row['forecast_date'].date()
            cursor.execute("""
                INSERT INTO forecast_results_rf (forecast_date, forecast_price, state_name, district_name, market_name, commodity_name) 
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (forecast_date, row['forecast_price'], selected_state, selected_district, selected_market, selected_commodity))

        
        conn.commit()

        # Calculate evaluation metrics
        y_pred = rf_model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = math.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        # Plot forecast
        fig = px.line(
            forecast_df, x='forecast_date', y='forecast_price', 
            title=f'Price Forecast for {selected_commodity} in {selected_market}, {selected_district}, {selected_state}',
            labels={'forecast_date': 'Date', 'forecast_price': 'Forecast Price'}
        )

        # Display evaluation metrics
        metrics = html.Div([
            html.H4('Forecast Evaluation Metrics:'),
            html.P(f"Mean Absolute Error (MAE): {mae:.2f}"),
            html.P(f"Mean Squared Error (MSE): {mse:.2f}"),
            html.P(f"Root Mean Squared Error (RMSE): {rmse:.2f}"),
            html.P(f"R-squared (R²): {r2:.2f}")
        ])

        return fig, metrics

# Heatmap visualization of average prices by year and month
@app.callback(
    Output('heatmap', 'figure'),
    [Input('state-dropdown', 'value'), 
     Input('district-dropdown', 'value'), 
     Input('market-dropdown', 'value'), 
     Input('commodity-dropdown', 'value')]
)
def update_heatmap(selected_state, selected_district, selected_market, selected_commodity):
    if None in [selected_state, selected_district, selected_market, selected_commodity]:
        return px.imshow([[0]], labels={'x': 'Month', 'y': 'Year', 'color': 'Price'})

    filtered_df = df[(df['state_name'] == selected_state) &
                     (df['district_name'] == selected_district) &
                     (df['market_name'] == selected_market) &
                     (df['commodity_name'] == selected_commodity)]

    heatmap_data = filtered_df.groupby(['year', 'month'])['modal_price'].mean().unstack()

    fig = px.imshow(
        heatmap_data,
        labels={'x': 'Month', 'y': 'Year', 'color': 'Average Price'},
        title=f'Average Monthly Prices for {selected_commodity} in {selected_market}, {selected_district}, {selected_state}'
    )
    return fig

# Run the Dash app
if __name__ == '__main__':
    app.run(debug=True)
