from dash import Dash, dcc, html
from dash.dependencies import Input, Output

app = Dash(__name__)

app.layout = html.Div(
    [
        dcc.Interval(
            id="interval-component", interval=1 * 100, n_intervals=0  # in milliseconds
        ),
        html.Div(id="live-update-text"),
    ]
)


@app.callback(
    Output("live-update-text", "children"), Input("interval-component", "n_intervals")
)
def update_layout(n):
    return "Crunching numbers: {}".format(n)


if __name__ == "__main__":
    app.run_server(debug=True)
