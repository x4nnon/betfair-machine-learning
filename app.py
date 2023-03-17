import os
import streamlit as st

from utils.run_btns import run_regressor_btn, run_rl_btn

st.set_page_config(layout="wide")


def set_row(entries: dict, max_per_row=4):
    num_entries = len(entries)
    num_rows = int(num_entries / max_per_row) + (num_entries % max_per_row > 0)
    rows = []
    entries = {k.replace("_", " ").title(): v for k, v in entries.items()}
    for i in range(num_rows):
        row = st.columns(min(max_per_row, num_entries - i * max_per_row))
        rows.append(row)
    for i, (key, val) in enumerate(entries.items()):
        row = rows[i // max_per_row]
        col = row[i % max_per_row]
        input_type = val.get("input_type", "text_input")
        disabled = val.get("disabled", False)

        if input_type == "text_input":
            entries[key]["value"] = col.text_input(
                key, val.get("default"), disabled=disabled
            )
        elif input_type == "selectbox":
            entries[key]["value"] = col.selectbox(
                key, val.get("options"), disabled=disabled
            )
        elif input_type == "checkbox":
            entries[key]["value"] = col.checkbox(
                key, val.get("default"), disabled=disabled
            )
        elif input_type == "number_input":
            entries[key]["value"] = col.number_input(
                key,
                val.get("min_value"),
                val.get("max_value"),
                val.get("default"),
                disabled=disabled,
            )

    return entries


# Create select boxes for the input parameters
with st.container():
    st.markdown("# Simulation")
    reg_tab, rl_tab = st.tabs(["Regression", "Reinforcement Learning"])
    env_input = {
        "Name": {
            "type": "text_input",
            "default": "Pre-live Horse Race",
            "disabled": True,
        },
        "Action Space": {
            "input_type": "number_input",
            "default": 3,
            "disabled": True,
        },
        "State Space": {
            "input_type": "number_input",
            "default": 10,
            "disabled": True,
        },
        "Reward": {"input_type": "number_input", "default": 1, "disabled": True},
    }

    reg_input = {
        "Strategy": {
            "default": "Strategy1",
            "input_type": "selectbox",
            "options": ["Strategy1", "Strategy2"],
        },
        "Test Folder Path": {
            "default": "horses_jul_wins",
            "options": ["horses_jul_wins"],
            "input_type": "selectbox",
        },
        "BSPs path": {
            "default": "july_22_bsps",
            "options": ["july_22_bsps"],
            "input_type": "selectbox",
        },
        "Model Name": {
            "default": "BayesianRidge",
            "input_type": "selectbox",
            "options": ["BayesianRidge", "Ridge"],
        },
    }
    reg_output = {}
    run_button = False
    with reg_tab:
        with st.form(key="reg_form"):
            reg_output = set_row(reg_input)

            max_races = len(os.listdir("horses_jul_wins"))
            races = st.number_input("Races", min_value=1, max_value=max_races)
            save = st.checkbox("Save", False)
            print(reg_output["Test Folder Path"])
            run_button = st.form_submit_button("Run Regressor Simulation")

        if run_button:
            run_regressor_btn(
                strategy=reg_output["Strategy"]["value"],
                test_folder_path=reg_output["Test Folder Path"]["value"],
                bsps_path=reg_output["BSPs path"]["value"],
                model_name=reg_output["Model Name"]["value"],
                races=races,
                save=save,
            )

    with rl_tab:
        with st.form(key="rl_form"):
            st.subheader("Environment")
            set_row(env_input)

            st.subheader("Model")
            st.selectbox("RL Model", ["PEP2", "TRPO"])

            hyperparams_input = {
                "n_steps": {
                    "input_type": "number_input",
                    "default": 128,
                    "min_value": 8,
                    "max_value": 4096,
                    "step": 8,
                },
                "ent_coef": {
                    "input_type": "number_input",
                    "default": 0.0,
                    "min_value": 0.0,
                    "max_value": 1.0,
                    "step": 0.01,
                },
                "learning_rate": {
                    "input_type": "number_input",
                    "default": 2.5e-4,
                    "min_value": 1e-5,
                    "max_value": 1e-3,
                    "step": 1e-5,
                },
                "vf_coef": {
                    "input_type": "number_input",
                    "default": 0.5,
                    "min_value": 0.0,
                    "max_value": 1.0,
                    "step": 0.01,
                },
                "max_grad_norm": {
                    "input_type": "number_input",
                    "default": 0.5,
                    "min_value": 0.0,
                    "max_value": 1.0,
                    "step": 0.01,
                },
                "lam": {
                    "input_type": "number_input",
                    "default": 0.95,
                    "min_value": 0.0,
                    "max_value": 1.0,
                    "step": 0.01,
                },
                "nminibatches": {
                    "input_type": "number_input",
                    "default": 4,
                    "min_value": 1,
                    "max_value": 64,
                    "step": 1,
                },
                "cliprange": {
                    "input_type": "number_input",
                    "default": 0.2,
                    "min_value": 0.0,
                    "max_value": 1.0,
                    "step": 0.01,
                },
                "cliprange_vf": {
                    "input_type": "number_input",
                    "min_value": 0.0,
                    "max_value": 1.0,
                    "step": 0.01,
                },
                "cliprange_feedback": {
                    "input_type": "number_input",
                    "min_value": 0.0,
                    "max_value": 1.0,
                    "step": 0.01,
                },
                "ent_coef_schedule": {"input_type": "text_input", "default": "linear"},
                "target_kl": {
                    "input_type": "number_input",
                    "min_value": 0.0,
                    "max_value": 10.0,
                    "step": 0.01,
                },
                "seed": {
                    "input_type": "number_input",
                    "default": 42,
                    "min_value": 0,
                    "max_value": 2**32 - 1,
                    "step": 1,
                },
                "use_sde": {"input_type": "checkbox", "default": False},
                "sde_sample_freq": {
                    "input_type": "number_input",
                    "default": -1,
                    "min_value": -1,
                },
            }
            hyperparams_output = set_row(hyperparams_input)
            # TODO fix model
            model = ""

            st.checkbox("Auto Hyperparameter Tuning", False)
            run_button = st.form_submit_button("Run Regressor Simulation")
            if run_button:
                run_rl_btn(model, save)
