from io import StringIO
import sys
from threading import Thread
import time
import streamlit as st
from flumine_simulator import piped_run
from flumine import clients

from onedrive import Onedrive
from utils.config import SITE_URL, app_principal
from utils.utils import get_simulation_plot


def rl_run(model_name, save):
    k = model_name
    return


def run_simulation(
    # console_output,
    strategy: str,
    onedrive: Onedrive,
    test_folder_path: str,
    bsps_path: str,
    model_name: str,
    races: int,
    save: bool,
):
    capture_output = StringIO()
    sys.stdout = capture_output

    # Run the piped_run function with the input parameters
    tracker = piped_run(
        strategy=strategy,
        onedrive=onedrive,
        client=clients.SimulatedClient(),
        test_folder_path=test_folder_path,
        bsps_path=bsps_path,
        model_name=model_name,
        races=races,
        save=save,
    )

    # Restore sys.stdout to the original value
    sys.stdout = sys.__stdout__

    # Display the simulation results
    st.markdown("## Simulation Results")
    st.write("Total expected profit is ", tracker["expected_profit_plotter"][-1])
    fig = get_simulation_plot(tracker, strategy)
    st.pyplot(fig)


def run_regressor_btn(
    strategy: str,
    test_folder_path: str,
    bsps_path: str,
    model_name: str,
    races: int,
    save: bool,
):
    # with st.spinner("Running simulation..."):
    console_output = st.empty()

    onedrive = Onedrive(
        client_id=app_principal["client_id"],
        client_secret=app_principal["client_secret"],
        site_url=SITE_URL,
    )
    # Start a new thread to run the simulation
    t = Thread(
        target=run_simulation,
        args=(
            # console_output,
            strategy,
            onedrive,
            test_folder_path,
            bsps_path,
            model_name,
            races,
            save,
        ),
    )

    t.start()

    # Display the progress of the simulation
    start_time = time.time()
    while t.is_alive():
        elapsed_time = time.time() - start_time
        console_output.text(f"Elapsed time: {elapsed_time:.2f} s\n")
        time.sleep(1)
    console_output.text("Simulation completed.\n")


def run_rl_btn(model, save):
    if st.button(f"Run RL Simulation"):
        with st.spinner("Running rl simulation..."):
            console_output = st.empty()

            t = Thread(
                target=rl_run,
                args=(
                    # console_output,
                    model,
                    save,
                ),
            )
            t.start()

            # Display the progress of the simulation
            start_time = time.time()
            while t.is_alive():
                elapsed_time = time.time() - start_time
                console_output.text(f"Elapsed time: {elapsed_time:.2f} s\n")
                time.sleep(1)
            console_output.text("Simulation completed.\n")
