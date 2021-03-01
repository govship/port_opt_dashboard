import numpy as np
import pandas as pd
import yfinance as yf
import warnings
import matplotlib.pyplot as plt
import riskfolio.RiskFunctions as rk
import plotly.graph_objects as go

warnings.filterwarnings("ignore")

yf.pdr_override()
pd.options.display.float_format = '{:.4%}'.format

# Date range
start = '2016-01-01'
end = '2019-12-30'

# Tickers of assets
assets = ['JCI', 'TGT', 'CMCSA', 'CPB', 'MO', 'NBL', 'APA', 'MMC', 'JPM',
          'ZION', 'PSA', 'BAX', 'BMY', 'LUV', 'PCAR', 'TXT', 'DHR',
          'DE', 'MSFT', 'HPQ', 'SEE', 'VZ', 'CNP', 'NI']
assets.sort()

# Downloading data
data = yf.download(assets, start = start, end = end)
data = data.loc[:,('Adj Close', slice(None))]
data.columns = assets


# Calculating returns

Y = data[assets].pct_change().dropna()


import riskfolio.Portfolio as pf

# Building the portfolio object
port = pf.Portfolio(returns=Y)
# Calculating optimum portfolio

# Select method and estimate input parameters:

method_mu='hist' # Method to estimate expected returns based on historical data.
method_cov='hist' # Method to estimate covariance matrix based on historical data.

port.assets_stats(method_mu=method_mu, method_cov=method_cov, d=0.94)

# Estimate optimal portfolio:

model='Classic' # Could be Classic (historical), BL (Black Litterman) or FM (Factor Model)
rm = 'MV' # Risk measure used, this time will be variance
obj = 'Sharpe' # Objective function, could be MinRisk, MaxRet, Utility or Sharpe
hist = True # Use historical scenarios for risk measures that depend on scenarios
rf = 0 # Risk free rate
l = 0 # Risk aversion factor, only useful when obj is 'Utility'

w = port.optimization(model=model, rm=rm, obj=obj, rf=rf, l=l, hist=hist)

print(w)
# w = w.reset_index
# print(w)

import riskfolio.PlotFunctions as plf

# Plotting the composition of the portfolio

# ax = plf.plot_pie(w=w, title='Sharpe Mean Variance', others=0.05, nrow=25, cmap = "tab20",
#                   height=6, width=10, ax=None)

# plt.show()

rm_names = [
    "Standard Deviation",
    "Mean Absolute Deviation",
    "Semi Standard Deviation",
    "Value at Risk",
    "Conditional Value at Risk",
    "Entropic Value at Risk",
    "Worst Realization",
    "First Lower Partial Moment",
    "Second Lower Partial Moment",
    "Max Drawdown",
    "Average Drawdown",
    "Drawdown at Risk",
    "Conditional Drawdown at Risk",
    "Ulcer Index",
]


rmeasures = [
    "MV",
    "MAD",
    "MSV",
    "VaR",
    "CVaR",
    "EVaR",
    "WR",
    "FLPM",
    "SLPM",
    "MDD",
    "ADD",
    "DaR",
    "CDaR",
    "UCI",
]


def plot_frontier(
    w_frontier,
    mu,
    cov=None,
    returns=None,
    rm="MV",
    rf=0,
    alpha=0.05,
    cmap="viridis",
    w=None,
    label="Portfolio",
    marker="*",
    s=16,
    c="r",
    height=6,
    width=10,
    ax=None,
):
    """
    Creates a plot of the efficient frontier for a risk measure specified by
    the user.

    Parameters
    ----------
    w_frontier : DataFrame
        Portfolio weights of some points in the efficient frontier.
    mu : DataFrame of shape (1, n_assets)
        Vector of expected returns, where n_assets is the number of assets.
    cov : DataFrame of shape (n_features, n_features)
        Covariance matrix, where n_features is the number of features.
    returns : DataFrame of shape (n_samples, n_features)
        Features matrix, where n_samples is the number of samples and
        n_features is the number of features.
    rm : str, optional
        The risk measure used to estimate the frontier.
        The default is 'MV'. Posible values are:

        - 'MV': Standard Deviation.
        - 'MAD': Mean Absolute Deviation.
        - 'MSV': Semi Standard Deviation.
        - 'FLPM': First Lower Partial Moment (Omega Ratio).
        - 'SLPM': Second Lower Partial Moment (Sortino Ratio).
        - 'CVaR': Conditional Value at Risk.
        - 'EVaR': Conditional Value at Risk.
        - 'WR': Worst Realization (Minimax)
        - 'MDD': Maximum Drawdown of uncompounded returns (Calmar Ratio).
        - 'ADD': Average Drawdown of uncompounded returns.
        - 'DaR': Drawdown at Risk of uncompounded returns.
        - 'CDaR': Conditional Drawdown at Risk of uncompounded returns.
        - 'UCI': Ulcer Index of uncompounded returns.

    rf : float, optional
        Risk free rate or minimum acceptable return. The default is 0.
    alpha : float, optional
        Significante level of VaR, CVaR, EVaR, DaR and CDaR.
        The default is 0.05.
    cmap : cmap, optional
        Colorscale, represente the risk adjusted return ratio.
        The default is 'viridis'.
    w : DataFrame of shape (n_assets, 1), optional
        A portfolio specified by the user. The default is None.
    label : str, optional
        Name of portfolio that appear on plot legend.
        The default is 'Portfolio'.
    marker : str, optional
        Marker of w_. The default is '*'.
    s : float, optional
        Size of marker. The default is 16.
    c : str, optional
        Color of marker. The default is 'r'.
    height : float, optional
        Height of the image in inches. The default is 6.
    width : float, optional
        Width of the image in inches. The default is 10.
    ax : matplotlib axis, optional
        If provided, plot on this axis. The default is None.

    Raises
    ------
    ValueError
        When the value cannot be calculated.

    Returns
    -------
    ax : matplotlib Axes
        Returns the Axes object with the plot for further tweaking.

    Example
    -------
    ::

        label = 'Max Risk Adjusted Return Portfolio'
        mu = port.mu
        cov = port.cov
        returns = port.returns

        ax = plf.plot_frontier(w_frontier=ws, mu=mu, cov=cov, returns=returns,
                               rm=rm, rf=0, alpha=0.05, cmap='viridis', w=w1,
                               label='Portfolio', marker='*', s=16, c='r',
                               height=6, width=10, ax=None)

    .. image:: images/MSV_Frontier.png

    """

    if not isinstance(w_frontier, pd.DataFrame):
        raise ValueError("w_frontier must be a DataFrame")

    if not isinstance(mu, pd.DataFrame):
        raise ValueError("mu must be a DataFrame")

    if not isinstance(cov, pd.DataFrame):
        raise ValueError("cov must be a DataFrame")

    if not isinstance(returns, pd.DataFrame):
        raise ValueError("returns must be a DataFrame")

    if returns.shape[1] != w_frontier.shape[0]:
        a1 = str(returns.shape)
        a2 = str(w_frontier.shape)
        raise ValueError("shapes " + a1 + " and " + a2 + " not aligned")

    if w is not None:
        if not isinstance(w, pd.DataFrame):
            raise ValueError("w must be a DataFrame")

        if w.shape[1] > 1 and w.shape[0] == 0:
            w = w.T
        elif w.shape[1] > 1 and w.shape[0] > 0:
            raise ValueError("w must be a column DataFrame")

        if returns.shape[1] != w.shape[0]:
            a1 = str(returns.shape)
            a2 = str(w.shape)
            raise ValueError("shapes " + a1 + " and " + a2 + " not aligned")


    mu_ = np.array(mu, ndmin=2)

    item = rmeasures.index(rm)
    x_label = rm_names[item] + " (" + rm + ")"
    # title = "Efficient Frontier Mean - " + x_label

    pretty_container_bgcolor = '#f9f9f9'
    fig = go.Figure(
        layout=go.Layout(
            # title=title,
            plot_bgcolor=pretty_container_bgcolor,
            hovermode='x',
            hoverdistance=100,
            spikedistance=1000,
            xaxis=dict(
                title=rm_names[item] + " (" + rm + ")",
                linecolor='#9a9a9a',
                showgrid=False,
                showspikes=True,
                spikethickness=3,
                spikedash='dot',
                spikecolor='#FF0000',
                spikemode='across'
            ),
            yaxis=dict(
                title="Expected Return",
                linecolor='#9a9a9a',
                showgrid=False
            )
        )
    )


    X1 = []
    Y1 = []
    Z1 = []

    for i in range(w_frontier.shape[1]):
        weights = np.array(w_frontier.iloc[:, i], ndmin=2).T
        risk = rk.Sharpe_Risk(
            weights, cov=cov, returns=returns, rm=rm, rf=rf, alpha=alpha
        )
        ret = mu_ @ weights
        ret = ret.item()
        ratio = (ret - rf) / risk

        X1.append(risk)
        Y1.append(ret)
        Z1.append(ratio)

    fig.add_trace(
        go.Scatter(
            x=X1,
            y=Y1
        )
    )

    # ax1 = ax.scatter(X1, Y1, c=Z1, cmap=cmap)

    if w is not None:
        X2 = []
        Y2 = []
        for i in range(w.shape[1]):
            weights = np.array(w.iloc[:, i], ndmin=2).T
            risk = rk.Sharpe_Risk(
                weights, cov=cov, returns=returns, rm=rm, rf=rf, alpha=alpha
            )
            ret = mu_ @ weights
            ret = ret.item()
            ratio = (ret - rf) / risk

            X2.append(risk)
            Y2.append(ret)

        fig.add_trace(
            go.Scatter(
                x=X2,
                y=Y2
            )
        )

    #
    # xmin = np.min(X1) - np.abs(np.max(X1) - np.min(X1)) * 0.1
    # xmax = np.max(X1) + np.abs(np.max(X1) - np.min(X1)) * 0.1
    # ymin = np.min(Y1) - np.abs(np.max(Y1) - np.min(Y1)) * 0.1
    # ymax = np.max(Y1) + np.abs(np.max(Y1) - np.min(Y1)) * 0.1
    #
    # ax.set_ylim(ymin, ymax)
    # ax.set_xlim(xmin, xmax)
    #
    # ax.set_yticklabels(["{:.4%}".format(x) for x in ax.get_yticks()])
    # ax.set_xticklabels(["{:.4%}".format(x) for x in ax.get_xticks()])
    #
    # ax.tick_params(axis="y", direction="in")
    # ax.tick_params(axis="x", direction="in")
    #
    # ax.grid(linestyle=":")
    #
    # colorbar = ax.figure.colorbar(ax1)
    # colorbar.set_label("Risk Adjusted Return Ratio")
    #
    # fig = plt.gcf()
    # fig.tight_layout()

    return fig.show()


points = 50 # Number of points of the frontier
frontier = port.efficient_frontier(model=model, rm=rm, points=points, rf=rf, hist=hist)
print(frontier)

label = 'Max Risk Adjusted Return Portfolio' # Title of point
mu = port.mu # Expected returns
cov = port.cov # Covariance matrix
returns = port.returns # Returns of the assets

ax = plot_frontier(w_frontier=frontier, mu=mu, cov=cov, returns=returns, rm=rm,
                       rf=rf, alpha=0.01, cmap='viridis', w=w, label=label,
                       marker='*', s=16, c='r', height=6, width=10, ax=None)

# fig.show()