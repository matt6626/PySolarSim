from multiprocessing import Pipe, connection

from dash import dash, dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go

import json


def gui(parent_conn: connection.Connection):

    # Connection pipe back to the parent process
    parent_conn = parent_conn

    # Create app
    app = dash.Dash()
    app.layout = html.Div(
        [
            dcc.Graph(id="live-graph"),
            dcc.Interval(id="data-stream", interval=100, n_intervals=0),
            dcc.Store(id="data-store"),
        ]
    )

    # note for implementation of data stream, only stream the new data coming in, store the history of graphs in the dcc store
    @app.callback(
        [Output("data-store", "data")],
        [Input("data-stream", "n_intervals")],
        [State("data-store", "data")],
    )
    def process_data_stream(n_intervals, data):
        if data is None:
            # Initialize with some default data
            data = {"t": [0], "vars": {}}
            # parent_conn.send("gui-ready")
            return [data]
            # return json.dumps(initial_data)  # Serialize the dictionary to a JSON string

        if parent_conn.poll():
            try:
                msg = parent_conn.recv()
            except:
                return dash.no_update
            # print(f"stream: {msg}")

            # store the new data to a running history
            data["t"].append(data["t"][-1] + msg["dt"])
            # print(f"keys: [{msg.items()}]")
            vars = data.get("vars", {})
            # print(f"vars: {vars}")
            for key, value in msg.get("vars", {}).items():
                # vars[key].append(value)
                if vars.get(key, None) is None:
                    vars[key] = [value]
                else:
                    vars[key].append(value)
            data["vars"] = vars

            return [data]  # Serialize the dictionary to a JSON string
        return dash.no_update

    # Create callbacks
    @app.callback(
        [Output("live-graph", "figure")],
        [Input("data-store", "data")],
    )
    def update_graph_scatter(
        data,
    ):
        # print(f"Processed data: {data}")
        state = data

        # print(f'state["dt"] {state["dt"]}')

        # Check if stored_data is None
        if data is None:
            pass
            # If it's None, initialize it with some default values
            # state = {"t": [0], "vars": {key: [0] for key in self.vars.keys()}}
            # state = {"t": [0], "vars": {"foo": [1], "bar": [1]}}

        # Create a new figure with the updated data
        fig = go.Figure()

        for key in state["vars"].keys():
            fig.add_trace(
                go.Scatter(
                    x=state["t"], y=state["vars"][key], mode="lines", name=f"{key}"
                )
            )

        # Get the min and max of all y values
        y_values = [value for sublist in state["vars"].values() for value in sublist]
        if len(y_values) != 0:
            y_min = min(y_values)
            y_max = max(y_values)
        else:
            y_min = 0
            y_max = 1

        fig.update_xaxes(range=[min(state["t"]), max(state["t"])])  # set x-axis range
        fig.update_yaxes(range=[y_min, y_max])  # set y-axis range

        fig.update_layout(
            height=600,
            width=600,
            title_text="Voltage Mode Controller",
            # transition=dict(duration=50, easing="cubic-in-out"),
        )

        return [fig]

    app.run_server(debug=True, use_reloader=False)
    parent_conn.close()
