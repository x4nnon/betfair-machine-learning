import streamlit as st
import yaml
from utils.utils import get_simulation_plot


def display_data_row_wise(
    data,
    num_columns,
    key_style="font-weight: bold; color: #666666",
    value_style="",
    row_height="1rem",
):
    # Split the data into rows to fit the grid layout
    data_rows = [
        list(data.items())[i : i + num_columns]
        for i in range(0, len(data), num_columns)
    ]

    # Display the data row-wise
    for row in data_rows:
        row_container = st.columns(num_columns)
        for col_idx, (key, value) in enumerate(row):
            with row_container[col_idx].container():
                # Set a fixed height for the row to align keys and values
                st.markdown(
                    f"<div style='height: {row_height};'></div>", unsafe_allow_html=True
                )
                # Write the key in bold and a light shade of gray
                st.markdown(
                    f"<p style='{key_style}'>{key}:</p>", unsafe_allow_html=True
                )
                # Write the value below the key
                st.markdown(
                    f"<p style='{value_style}'>{value}</p>", unsafe_allow_html=True
                )


# Display the data table
tracker = dict()
with open("dummy_data/tracker.yaml", "r") as f:
    tracker = yaml.safe_load(f)

# Extract the data that you want to display
data_to_display = {
    "Total Profit": tracker["total_profit"],
    "Total Matched Correct": tracker["total_matched_correct"],
    "Total Matched Incorrect": tracker["total_matched_incorrect"],
    "Total Back Matched Correct": tracker["total_back_matched_correct"],
    "Total Back Matched Incorrect": tracker["total_back_matched_incorrect"],
    "Total Lay Matched Correct": tracker["total_lay_matched_correct"],
    "Total Lay Matched Incorrect": tracker["total_lay_matched_incorrect"],
    "Total Matched Correct Margin": tracker["total_m_c_marg"],
    "Total Matched Incorrect Margin": tracker["total_m_i_marg"],
    "Total Green Margin": tracker["total_green_margin"],
    "Total Amount Gambled": tracker["total_amount_gambled"],
    "Race Counter": tracker["race_counter"],
    "Total Q Correct": tracker["total_q_correct"],
    "Total Q Incorrect": tracker["total_q_incorrect"],
    "Total Matched Correct Margin": tracker["total_m_correct_margin"],
    "Total Matched Incorrect Margin": tracker["total_m_incorrect_margin"],
    "Total Q Margin": tracker["total_q_margin"],
}
# Display the data in the app
st.title("Simulation Data")

col1, col2 = st.columns([1, 2])

# Display the data in the first column
with col1:
    display_data_row_wise(data=data_to_display, num_columns=4)

    # Display the plot in the second column
with col2:
    fig = get_simulation_plot(tracker, "Strategy1", fig_size=(8, 5))
    st.pyplot(fig=fig)
