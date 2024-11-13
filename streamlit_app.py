import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Load and preprocess the dataset
def train_model():
    df = pd.read_csv("premier_league_2223.csv")
    
    # Check the columns to make sure they match the expected number
    print("Columns in dataset:", df.columns)
    
    # Filter and rename columns (make sure these columns exist in the dataset)
    df = df[['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'HS', 'AS', 'HST', 'AST', 'HF', 'AF', 'HC', 'AC', 'HY', 'AY', 'HR', 'AR']]
    
    # Renaming columns
    df.columns = ['HomeTeam', 'AwayTeam', 'HomeGoals', 'AwayGoals', 'Result', 'HomeShots', 'AwayShots', 
                  'HomeShotsOnTarget', 'AwayShotsOnTarget', 'HomeFouls', 'AwayFouls', 'HomeCorners', 'AwayCorners', 
                  'HomeYellowCards', 'AwayYellowCards', 'HomeRedCards', 'AwayRedCards']
    
    # Convert categorical variables (teams) to numeric codes
    df['HomeTeam'] = df['HomeTeam'].astype('category').cat.codes
    df['AwayTeam'] = df['AwayTeam'].astype('category').cat.codes
    df['Result'] = df['Result'].map({'H': 1, 'D': 0, 'A': -1})  # Map home win, draw, away win to numerical values
    
    # Fill missing values
    df = df.fillna(df.mean())
    
    # Features (X) and target variable (y)
    X = df[['HomeTeam', 'AwayTeam', 'HomeShots', 'AwayShots', 'HomeShotsOnTarget', 'AwayShotsOnTarget', 'HomeFouls', 'AwayFouls', 
            'HomeCorners', 'AwayCorners', 'HomeYellowCards', 'AwayYellowCards', 'HomeRedCards', 'AwayRedCards']]
    
    y = df['Result']
    
    # Train the model (Random Forest)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

# Train the model (or load it from a saved file)
model = train_model()

# Streamlit interface
st.title('Premier League Match Outcome Prediction')

# User input for home team and away team
home_team = st.selectbox('Select Home Team', ['Manchester United', 'Liverpool', 'Chelsea', 'Arsenal', 'Manchester City', 'Tottenham Hotspur', 'Other'])
away_team = st.selectbox('Select Away Team', ['Manchester United', 'Liverpool', 'Chelsea', 'Arsenal', 'Manchester City', 'Tottenham Hotspur', 'Other'])

# Collecting match stats (can be simplified, or more stats can be added)
home_shots = st.slider('Home Team Shots', 0, 50, 15)
away_shots = st.slider('Away Team Shots', 0, 50, 15)
home_shots_on_target = st.slider('Home Team Shots on Target', 0, 20, 5)
away_shots_on_target = st.slider('Away Team Shots on Target', 0, 20, 5)
home_fouls = st.slider('Home Team Fouls', 0, 30, 5)
away_fouls = st.slider('Away Team Fouls', 0, 30, 5)
home_corners = st.slider('Home Team Corners', 0, 20, 5)
away_corners = st.slider('Away Team Corners', 0, 20, 5)
home_yellow_cards = st.slider('Home Team Yellow Cards', 0, 5, 1)
away_yellow_cards = st.slider('Away Team Yellow Cards', 0, 5, 1)
home_red_cards = st.slider('Home Team Red Cards', 0, 5, 0)
away_red_cards = st.slider('Away Team Red Cards', 0, 5, 0)

# Prepare features for prediction
home_team_code = {'Manchester United': 0, 'Liverpool': 1, 'Chelsea': 2, 'Arsenal': 3, 'Manchester City': 4, 'Tottenham Hotspur': 5, 'Other': 6}
away_team_code = {'Manchester United': 0, 'Liverpool': 1, 'Chelsea': 2, 'Arsenal': 3, 'Manchester City': 4, 'Tottenham Hotspur': 5, 'Other': 6}

home_team_code = home_team_code.get(home_team, 6)
away_team_code = away_team_code.get(away_team, 6)

# Prepare the input features
input_features = pd.DataFrame([[home_team_code, away_team_code, home_shots, away_shots, home_shots_on_target, away_shots_on_target, home_fouls, away_fouls, 
                                home_corners, away_corners, home_yellow_cards, away_yellow_cards, home_red_cards, away_red_cards]])

# Make prediction
prediction = model.predict(input_features)

# Show prediction result
if prediction == 1:
    st.write("Prediction: Home Team Wins!")
elif prediction == 0:
    st.write("Prediction: Draw!")
else:
    st.write("Prediction: Away Team Wins!")
