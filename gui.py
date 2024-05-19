from multiprocessing import Queue, Lock, Semaphore
import queue

from dash import dash, dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go

from remote_pdb import RemotePdb

import logging

logging.basicConfig(level=logging.INFO)

def gui(to_gui_queue: Queue, from_gui_queue: Queue):

    # TODO: add init to DCC Store to avoid the requirement of the nonlocal keyword in the data_stream callback
    initialised = Semaphore()

    # Create app
    app = dash.Dash(__name__, title="Simulation", update_title=None)
    app.layout = html.Div(
        [
            dcc.Graph(id="live-graph"),
            # TODO: investigate data stores not updating when the callback interval is short (eg. 100 ms)
            dcc.Interval(id="data-stream", interval=10000, n_intervals=0),
            dcc.Store(id="interval-count", clear_data=True),
            dcc.Store(id="data-store", clear_data=True),
            dcc.Store(id="dummy-output"),
        ]
    )

    data_stream_lock = Lock()

    def append_new_data_to_store(store, msg):
        # print("START: append_new_data_to_store")

        # Append the time step based on delta time received
        prev_time = store["t"][-1]
        # print(f'prev_store["t"]: {store["t"]}')
        dt = msg["dt"]

        if prev_time is None or dt is None:
            print("error")
            print(f"prev_time: {prev_time}")
            print(f"dt: {dt}")

        curr_time = prev_time + dt
        # print(f"prev_time: {prev_time*10**6}")

        # print(f"dt: {dt}")
        # print(f"curr_time: {curr_time*10**6}")
        store["t"].append(curr_time)
        # print(f'post_store["t"]: {store["t"]}')

        # Append the data from any variable received
        store_vars = store.get("vars", {})  # Copy of existing vars history
        recvd_vars = msg.get("vars", {})
        for key, value in recvd_vars.items():
            if store_vars.get(key, None) is None:
                store_vars[key] = [value]
            else:
                store_vars[key].append(value)
        store["vars"] = store_vars

        recvd_time = recvd_vars["t"]

        if curr_time != recvd_time:
            # print(f"curr_time: {curr_time}")
            # print(f'vars["t"]: {vars["t"]}')
            raise Exception(
                f"curr_time: {curr_time} does not equal recvd_time: {recvd_time}"
            )
        # print(f"END: append_new_data_to_store\n")

    # this callback guarantees that the data-store is updated before the lock is released
    @app.callback(Output("dummy-output", "data"), [Input("data-store", "data")])
    def new_data_callback(data):
        logging.info("Entered new_data_callback")
        try:
            print("Release lock")
            data_stream_lock.release()
        except Exception as e:
            logging.error(f"Error in new_data_callback: {e}")
        finally:
            logging.info("Exiting new_data_callback")
        return None

    # note for implementation of data stream, only stream the new data coming in, store the history of graphs in the dcc store
    @app.callback(
        [Output("data-store", "data"), Output("interval-count", "data")],
        [Input("data-stream", "n_intervals")],
        [State("data-store", "data"), State("interval-count", "data")],
    )
    def process_data_stream(n_intervals, data, interval_count):
        logging.info("Entered process_data_stream")
        try:

            def int_print(str):
                print(f"n_int={interval_count}: {str}")

            def print_data(data: dict = {}):
                vars: dict = data.get("vars", None)
                for var_name, var in vars.items():
                    int_print(f"{var_name}_len={len(var)}")

            if not data_stream_lock.acquire(block=False):
                print(f"Failed attempt to acquire lock.")
                return [dash.no_update, interval_count]

            # RemotePdb("127.0.0.1", 4444).set_trace()
            if initialised.acquire(block=False):
                interval_count = 0
                int_print(f"PROCESS_DATA_STREAM: INIT")
                data = {"t": [0], "vars": {}}
                from_gui_queue.put("gui-ready")
                print_data(data)
                int_print(f"END: process_data_stream [init graph]")
                return [data, interval_count]

            if interval_count is None:
                print("err")
            interval_count = interval_count + 1

            int_print("")

            msg = None
            while not to_gui_queue.empty():
                try:
                    msg = to_gui_queue.get_nowait()
                    # print(f"n_intervals: {n_intervals}")
                    append_new_data_to_store(data, msg)
                except queue.Empty:
                    break
                except Exception as e:
                    print_data(data)
                    int_print(f"exception: {e} on {msg}")
                    int_print(f"END: process_data_stream")
                    data_stream_lock.release()
                    return [dash.no_update, interval_count]

            if msg is not None:
                print_data(data)
                int_print(f"END: process_data_stream [update graph]")
                int_print(f"{interval_count}")
                return [data, interval_count]
                int_print("foo")
            else:
                int_print(f"END: process_data_stream [no graph update]")
                data_stream_lock.release()
                return [dash.no_update, interval_count]
            int_print("bar")
        except Exception as e:
            logging.error(f"Error in process_data_stream: {e}")
        finally:
            logging.info("Exiting process_data_stream")

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
        # if data is None:
        # pass
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

    app.run_server(debug=False, use_reloader=False)
