import dash
import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc
from dash.dependencies import State, Input, Output
import matplotlib.pyplot as plt

import riskfolio.Portfolio as pf
import riskfolio.PlotFunctions as plf
import pandas as pd
import datetime
import yfinance as yf
import warnings

import plotly.graph_objects as go
import plot_functions as pfunc


warnings.filterwarnings("ignore")

yf.pdr_override()
pd.options.display.float_format = '{:.4%}'.format

# TODO: put list of assets into separate file
assets = [
"BCOIX",
"DODFX",
"GSSUX",
"AIIEX",
"MEIKX",
"VEXRX",
"VAIPX",
"VINIX",
"VMGMX",
"VIMAX",
"VGSLX",
"VSMAX",
"VBTLX",
"VTIAX",
"VWENX"
]

assets.sort()

models = ['Classic'] # "BL", FM]
risk_measures = ["MV", "MAD", "MSV", "FLPM", "SLPM", "CVaR", "WR", "MDD", "ADD", "CDaR", "UCI"]
objectives = ["MinRisk", "Utility", "Sharpe", "MaxRet"]

default_layout = dict(
    autosize=True,
    automargin=True,
    margin=dict(b=40, t=20),
    font=dict(
        color="#9fa6b7"
    ),
    plot_bgcolor="#f5f7fa",
    paper_bgcolor="#f5f7fa",
    legend=dict(
        font=dict(size=10),
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ),
)

data = yf.download(assets)
data = data.loc[:, ('Adj Close', slice(None))]
data.columns = assets
earliest_data_date = data.dropna().index.strftime('%Y-%m-%d')[0]
latest_data_date = data.dropna().index.strftime('%Y-%m-%d')[-1]
data = yf.download(assets, start=earliest_data_date, end=latest_data_date)


app = dash.Dash(
    __name__,
    meta_tags=[
        {
            "name": "viewport",
            "content": "width=device-width, initial-scale=1, maximum-scale=1.0, user-scalable=no",
        }
    ],
    # TODO: check meta_tags documentation
    external_stylesheets=[dbc.themes.GRID]
)

server = app.server

app.layout = html.Div(
    children=[
        html.Div(
            className="banner",
            children=[
                html.H6("Portfolio Optimizer"),
                html.Div(
                    html.A(
                        "Github",
                        target="_blank",
                        className="button",
                    ),
                    # TODO: Add href link to exchange
                ),
            ],
        ),
        dbc.Row(
            children=[
                dbc.Col(
                    lg=2, sm=12,
                    children=[
                        html.Div(
                            className="pretty_container",
                            children=[
                                html.Div(
                                    className="banner",
                                    style={"marginBottom": "1.5em"},
                                    children=[
                                        html.H6("Settings")
                                    ],
                                ),
                                html.Label("Assets:"),
                                dcc.Dropdown(
                                    # className="dropdown_period",
                                    id="assets_dropdown",
                                    options=[{"label": i, "value": i} for i in assets],
                                    value=assets,
                                    clearable=True,
                                    searchable=True,
                                    placeholder="Select assets",
                                    multi=True,
                                    style={"marginBottom": "1.5em"}
                                ),
                                html.Label("Date Range:"),
                                dcc.DatePickerRange(
                                    id="date_picker_range",
                                    min_date_allowed=datetime.datetime.strptime(earliest_data_date, '%Y-%m-%d'),
                                    max_date_allowed=datetime.datetime.strptime(latest_data_date, '%Y-%m-%d'),
                                    start_date=datetime.datetime.strptime(earliest_data_date, '%Y-%m-%d'),
                                    end_date=datetime.datetime.strptime(latest_data_date, '%Y-%m-%d'),
                                    style={
                                        "marginBottom": "1.5em",
                                        # "background-color": "#22252b",
                                        # "border-bottom-color": "#22252b",
                                        # "border-color": "#22252b",
                                        # "border-left-color": "#22252b",
                                        # "border-right-color": "#22252b",
                                        # "border-top-color": "#22252b",
                                        # "color": "#22252b",
                                        # "column-rule-color": "#22252b",
                                        # "outline-color": "#22252b",
                                        # "text-decoration-color": "#22252b",
                                    }
                                ),
                                html.Label("Model:"),
                                dcc.Dropdown(
                                    # className="dropdown_model",
                                    id="model_dropdown",
                                    options=[{"label": i, "value": i} for i in models],
                                    value=models[0],
                                    searchable=False,
                                    placeholder="Select a model",
                                    clearable=False,
                                    style={"marginBottom": "1.5em"}
                                ),
                                html.Label("Risk Measure:"),
                                dcc.Dropdown(
                                    # className="dropdown_model",
                                    id="risk_measure_dropdown",
                                    options=[{"label": i, "value": i} for i in risk_measures],
                                    value=risk_measures[0],
                                    searchable=False,
                                    placeholder="Select a risk measure",
                                    clearable=False,
                                    style={"marginBottom": "1.5em"}
                                ),
                                html.Label("Objective Function:"),
                                dcc.Dropdown(
                                    # className="dropdown_model",
                                    options=[{"label": i, "value": i} for i in objectives],
                                    value=objectives[0],
                                    id="objective_function_dropdown",
                                    searchable=False,
                                    placeholder="Select a objective",
                                    clearable=False,
                                    style={"marginBottom": "1.5em"}
                                ),
                                html.Div(
                                    dbc.Row(
                                        children=[
                                            dbc.Col(
                                                dbc.FormGroup(
                                                    children=[
                                                        html.Label("Risk Free Rate:"),
                                                        dbc.Input(
                                                            id="risk_free_rate_input",
                                                            type="number",
                                                            placeholder=0,
                                                            bs_size="sm",
                                                            size=10,
                                                        ),
                                                    ],
                                                ),
                                            ),
                                            dbc.Col(
                                                dbc.FormGroup(
                                                    children=[
                                                        html.Label("Risk Aversion Factor:"),
                                                        dbc.Input(
                                                            id="risk_aversion_factor_input",
                                                            type="number",
                                                            placeholder=0,
                                                            bs_size="sm",
                                                            size=10,
                                                        ),
                                                    ],
                                                ),
                                            ),
                                        ],
                                    ),
                                    style={"width": "100%", "marginBottom": "3em"},
                                ),
                            ],
                        ),
                    ],
                ),
                dbc.Col(
                    lg=10, sm=12,
                    children=[
                        html.Div(
                            className="pretty_container",
                            children=[
                                html.Div(
                                    className="banner",
                                    style={
                                        "marginBottom": "1.5em"
                                    },
                                    children=[
                                        html.H6("Optimized Portfolio")
                                    ],
                                ),
                                dbc.Row(
                                    children=[
                                        dbc.Col(
                                            dcc.Graph(
                                                id="asset_pie_chart",
                                                config={'displayModeBar': False, 'showTips': False}
                                            ),
                                            style={"background-color": "#f2f2f2", "color": "#f2f2f2"},
                                            width=8,
                                        ),
                                        dbc.Col(
                                            dcc.Graph(
                                                id="efficient_frontier_graph",
                                                config={'displayModeBar': False, 'showTips': False},
                                            ),
                                            width=4,
                                        ),
                                    ],
                                ),
                            ],
                        ),
                        html.Div(
                            className="pretty_container",
                            children=[
                                html.Div(
                                    className="banner",
                                    style={
                                        "marginBottom": "1.5em"
                                    },
                                    children=[
                                        html.H6("Performance vs Benchmark")
                                    ],
                                ),
                                dcc.Loading(
                                    id="benchmark_overlay_graph_loading",
                                    type="default",
                                    children=[
                                        dcc.Graph(
                                            id="benchmark_overlay_graph",
                                            figure={
                                                "layout": default_layout
                                            },
                                            config={'displayModeBar': False, 'showTips': False}
                                        ),
                                    ],
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        ),
    ],
)

@app.callback(
    # Output('asset_pie_chart', 'figure'),
    [
        Output('asset_pie_chart', 'figure'),
        Output('efficient_frontier_graph', 'figure'),
        Output('benchmark_overlay_graph', 'figure'),
    ],
    [
        Input('model_dropdown', 'value'),
        Input('risk_measure_dropdown', 'value'),
        Input('objective_function_dropdown', 'value'),
        Input('risk_free_rate_input', 'value'),
        Input('risk_aversion_factor_input', 'value'),
    ]
)

def portfolio_allocation(model, rm, obj, rfr, raf):
    data = yf.download(assets, start=earliest_data_date, end=latest_data_date)
    data = data.loc[:, ('Adj Close', slice(None))]
    data.columns = assets

    Y = data[assets].pct_change().dropna()

    port = pf.Portfolio(returns=Y)

    method_mu = "hist"  # Method to estimate expected returns based on historical data.
    method_cov = "hist"  # Method to estimate covariance matrix based on historical data.

    if model == "Classic":
        port.assets_stats(method_mu=method_mu, method_cov=method_cov, d=0.94)

    model = model  # Could be Classic (historical), BL (Black Litterman) or FM (Factor Model)
    rm = rm  # Risk measure used, this time will be variance
    obj = obj  # Objective function, could be MinRisk, MaxRet, Utility or Sharpe
    hist = True  # Use historical scenarios for risk measures that depend on scenarios
    rf = 0  # Risk free rate
    l = 0  # Risk aversion factor, only useful when obj is 'Utility'

    w = port.optimization(model=model, rm=rm, obj=obj, rf=rf, l=l, hist=hist)

    # points = 100
    # frontier = port.efficient_frontier(model=model, rm=rm, points=points, rf=rf, hist=hist)
    # label = 'Max Risk Adjusted Return Portfolio'
    # mu = port.mu
    # cov = port.cov
    # returns = port.returns

    fig = go.Figure(
        data=[
            go.Pie(
                labels=w.index,
                values=w["weights"]
            )
        ]
    )

    fig.update_layout(
        paper_bgcolor="#f2f2f2",
        autosize=False,
        width=680,
        height=680,
        margin=dict(
            l=20,
            r=20,
            b=50,
            t=50,
            pad=4
        ),
    )

    points = 50  # Number of points of the frontier
    frontier = port.efficient_frontier(model=model, rm=rm, points=points, rf=rf, hist=hist)

    label = 'Max Risk Adjusted Return Portfolio'  # Title of point
    mu = port.mu  # Expected returns
    cov = port.cov  # Covariance matrix
    returns = port.returns  # Returns of the assets

    ax = pfunc.plot_frontier(w_frontier=frontier, mu=mu, cov=cov, returns=returns, rm=rm,
                          rf=rf, alpha=0.01, cmap='viridis', w=w, label=label,
                          marker='*', s=16, c='r', height=6, width=10, ax=None)

    return fig, ax



if __name__ == "__main__":
    app.run_server(debug=True)