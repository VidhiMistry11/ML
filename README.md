import numpy as np
import ipywidgets as widgets
from IPython.display import display

import joblib

# Load the model from the file
model = joblib.load('random_forest_model.pkl') # Update the path as needed

# Define the unique values for dropdowns
venues = df['venue'].unique().tolist()
bat_teams = df['batting_team'].unique().tolist()
bowl_teams = df['bowling_team'].unique().tolist()

# Create UI elements with original values
venue_widget = widgets.Dropdown(
    options=venues,
    description='Venue:',
)

bat_team_widget = widgets.Dropdown(
    options=bat_teams,
    description='Bat Team:',
)

bowl_team_widget = widgets.Dropdown(
    options=bowl_teams,
    description='Bowl Team:',
)

overs_widget = widgets.IntSlider(
    value=1,
    min=1,
    max=50,
    step=1,
    description='Overs:',
)

wickets_widget = widgets.IntSlider(
    value=0,
    min=0,
    max=10,
    step=1,
    description='Wickets:',
)

runs_widget = widgets.IntText(
    value=0,
    description='Runs:',
)

runs_last_5_widget = widgets.IntText(
    value=0,
    description='Runs (last 5):',
)

wickets_last_5_widget = widgets.IntText(
    value=0,
    description='Wickets (last 5):',
)

output_widget = widgets.Output()

def get_dummy_columns():
    # Get dummy columns from the training data (X)
    return X.columns.tolist()

def predict_score(change):
    dummy_columns = get_dummy_columns()
    example_input_df = pd.DataFrame(np.zeros((1, len(dummy_columns))), columns=dummy_columns)

    # Set values for numerical features
    example_input_df['overs'] = overs_widget.value
    example_input_df['wickets'] = wickets_widget.value
    example_input_df['runs'] = runs_widget.value
    example_input_df['runs_last_5'] = runs_last_5_widget.value
    example_input_df['wickets_last_5'] = wickets_last_5_widget.value

    # Set values for categorical features (one-hot encoding)
    venue_column = f'venue_{venue_widget.value}'
    bat_team_column = f'bat_team_{bat_team_widget.value}'
    bowl_team_column = f'bowl_team_{bowl_team_widget.value}'

    # Check if columns exist before setting values
    if venue_column in dummy_columns:
        example_input_df[venue_column] = 1
    if bat_team_column in dummy_columns:
        example_input_df[bat_team_column] = 1
    if bowl_team_column in dummy_columns:
        example_input_df[bowl_team_column] = 1

    predicted_score = model.predict(example_input_df)

    with output_widget:
        output_widget.clear_output()
        display(f"Predicted Score: {predicted_score[0]:.2f}")

# Create a button to trigger prediction
predict_button = widgets.Button(description='Predict Score')
predict_button.on_click(predict_score)

# Display UI elements
display(venue_widget, bat_team_widget, bowl_team_widget, overs_widget, wickets_widget, runs_widget, runs_last_5_widget, wickets_last_5_widget, predict_button,output_widget )


