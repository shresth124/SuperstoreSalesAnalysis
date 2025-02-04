import pandas as pd
import dash
from dash import dcc, html
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.seasonal import seasonal_decompose

# Load the dataset
df = pd.read_csv("superstore.csv", parse_dates=['Order Date'], encoding='latin1')

# Data Preprocessing
df = df.dropna()  # Remove missing values
df['Year'] = df['Order Date'].dt.year  # Extract year for time-based analysis
df['Month'] = df['Order Date'].dt.month  # Extract month for detailed trend analysis

# Group data by Year and Month to aggregate sales for decomposition
df_monthly_sales = df.groupby(['Year', 'Month'])['Sales'].sum().reset_index()
df_monthly_sales['Date'] = pd.to_datetime(df_monthly_sales[['Year', 'Month']].assign(DAY=1))

# Perform Seasonal Decomposition using STL
decomposition = seasonal_decompose(df_monthly_sales['Sales'], model='additive', period=12)

# Get Trend, Seasonal, and Residual components
trend = decomposition.trend.dropna()
seasonal = decomposition.seasonal.dropna()
residual = decomposition.resid.dropna()

# Initialize the Dash app
app = dash.Dash(__name__)

# Sales by Category
fig_category = px.bar(df, x='Category', y='Sales', title='Sales by Category', color='Category', 
                      color_discrete_sequence=px.colors.qualitative.Set3)

# Profit by Region
fig_region = px.bar(df, x='Region', y='Profit', title='Profit by Region', color='Region',
                    color_discrete_sequence=px.colors.qualitative.Pastel)

# Sales Trend Over Time
fig_trend = px.line(df.groupby('Year')['Sales'].sum().reset_index(), x='Year', y='Sales', title='Sales Trend Over Years',
                    line_shape='linear', markers=True, template='plotly_dark')

# Sales by Sub-Category
fig_subcategory = px.bar(df, x='Sub-Category', y='Sales', title='Sales by Sub-Category', color='Sub-Category', 
                         color_discrete_sequence=px.colors.sequential.Plasma)

# Profit vs. Sales (Scatter Plot)
fig_profit_sales = px.scatter(df, x='Sales', y='Profit', title='Profit vs. Sales', color='Category', 
                              size='Quantity', hover_data=['Product Name'], color_discrete_sequence=px.colors.qualitative.Set1)

# Top 10 Products by Sales
top_products = df.groupby('Product Name')['Sales'].sum().reset_index().sort_values(by='Sales', ascending=False).head(10)
fig_top_products = px.bar(top_products, x='Product Name', y='Sales', title='Top 10 Products by Sales',
                          color='Product Name', color_discrete_sequence=px.colors.qualitative.Vivid)

# Order Quantity Distribution (Histogram)
fig_quantity_dist = px.histogram(df, x='Quantity', nbins=30, title='Order Quantity Distribution', 
                                 color_discrete_sequence=px.colors.qualitative.G10)

# Heatmap for Sales by Region and Category
sales_region_category = df.groupby(['Region', 'Category'])['Sales'].sum().reset_index()
fig_heatmap = px.density_heatmap(sales_region_category, x='Region', y='Category', z='Sales', 
                                 title='Sales Heatmap by Region and Category', color_continuous_scale='Viridis')

# Sales Trend by Month
monthly_sales = df.groupby(['Year', 'Month'])['Sales'].sum().reset_index()
fig_monthly_trend = px.line(monthly_sales, x='Month', y='Sales', color='Year', title='Sales Trend by Month',
                            line_shape='spline', markers=True, template='plotly_dark')

# Trend, Seasonality, and Residuals Plots
fig_trend_plot = go.Figure()
fig_trend_plot.add_trace(go.Scatter(x=df_monthly_sales['Date'][len(df_monthly_sales)-len(trend):], y=trend, 
                                   mode='lines', name='Trend', line=dict(color='royalblue', width=4)))
fig_trend_plot.update_layout(title='Trend Component', xaxis_title='Date', yaxis_title='Sales', 
                             template='plotly_dark', plot_bgcolor='rgba(0,0,0,0)')

fig_seasonal_plot = go.Figure()
fig_seasonal_plot.add_trace(go.Scatter(x=df_monthly_sales['Date'][len(df_monthly_sales)-len(seasonal):], y=seasonal, 
                                      mode='lines', name='Seasonality', line=dict(color='green', width=4)))
fig_seasonal_plot.update_layout(title='Seasonal Component', xaxis_title='Date', yaxis_title='Sales', 
                                template='plotly_dark', plot_bgcolor='rgba(0,0,0,0)')

fig_residual_plot = go.Figure()
fig_residual_plot.add_trace(go.Scatter(x=df_monthly_sales['Date'][len(df_monthly_sales)-len(residual):], y=residual, 
                                      mode='lines', name='Residuals', line=dict(color='red', width=4)))
fig_residual_plot.update_layout(title='Residual Component', xaxis_title='Date', yaxis_title='Sales', 
                                template='plotly_dark', plot_bgcolor='rgba(0,0,0,0)')

# Define the layout of the app with improved structure
app.layout = html.Div(style={'backgroundColor': '#1a1a1a', 'padding': '20px'}, children=[
    html.H1(children='Superstore Sales Dashboard', style={'textAlign': 'center', 'color': 'white', 'font-family': 'Arial', 'font-size': '36px'}),
    
    html.Div(style={'display': 'flex', 'flexWrap': 'wrap', 'gap': '20px', 'justifyContent': 'center'}, children=[
        html.Div(style={'width': '48%'}, children=[
            dcc.Graph(id='sales-by-category', figure=fig_category)
        ]),
        html.Div(style={'width': '48%'}, children=[
            dcc.Graph(id='profit-by-region', figure=fig_region)
        ])
    ]),

    html.Div(style={'display': 'flex', 'flexWrap': 'wrap', 'gap': '20px', 'justifyContent': 'center'}, children=[
        html.Div(style={'width': '48%'}, children=[
            dcc.Graph(id='sales-trend', figure=fig_trend)
        ]),
        html.Div(style={'width': '48%'}, children=[
            dcc.Graph(id='sales-by-subcategory', figure=fig_subcategory)
        ])
    ]),

    html.Div(style={'display': 'flex', 'flexWrap': 'wrap', 'gap': '20px', 'justifyContent': 'center'}, children=[
        html.Div(style={'width': '48%'}, children=[
            dcc.Graph(id='profit-vs-sales', figure=fig_profit_sales)
        ]),
        html.Div(style={'width': '48%'}, children=[
            dcc.Graph(id='top-products', figure=fig_top_products)
        ])
    ]),

    html.Div(style={'display': 'flex', 'flexWrap': 'wrap', 'gap': '20px', 'justifyContent': 'center'}, children=[
        html.Div(style={'width': '48%'}, children=[
            dcc.Graph(id='order-quantity-dist', figure=fig_quantity_dist)
        ]),
        html.Div(style={'width': '48%'}, children=[
            dcc.Graph(id='sales-heatmap', figure=fig_heatmap)
        ])
    ]),

    html.Div(style={'display': 'flex', 'flexWrap': 'wrap', 'gap': '20px', 'justifyContent': 'center'}, children=[
        html.Div(style={'width': '48%'}, children=[
            dcc.Graph(id='monthly-sales-trend', figure=fig_monthly_trend)
        ]),
        html.Div(style={'width': '48%'}, children=[
            dcc.Graph(id='trend-component', figure=fig_trend_plot)
        ])
    ]),

    html.Div(style={'display': 'flex', 'flexWrap': 'wrap', 'gap': '20px', 'justifyContent': 'center'}, children=[
        html.Div(style={'width': '48%'}, children=[
            dcc.Graph(id='seasonality-component', figure=fig_seasonal_plot)
        ]),
        html.Div(style={'width': '48%'}, children=[
            dcc.Graph(id='residual-component', figure=fig_residual_plot)
        ])
    ])
])

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
