import dash
import dash_bootstrap_components as dbc
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Load the datasets
matches_df = pd.read_csv("IPL_Matches_2022.csv")
ball_by_ball_df = pd.read_csv("IPL_Ball_by_Ball_2022.csv")

# Preprocess Data
matches_df['Date'] = pd.to_datetime(matches_df['Date'])  # Convert date to datetime format

# Initialize Dash App with Bootstrap theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# --- Dashboard Layout ---
app.layout = dbc.Container([
    html.H1("IPL 2022 Dashboard", className="text-center text-primary mb-4"),
    
    # Match Summary Section
    html.H2("Match Summary", className="text-center text-info"),
    dcc.Dropdown(
        id='match-dropdown',
        options=[{'label': f"{row['Team1']} vs {row['Team2']} ({row['Date'].strftime('%Y-%m-%d')})", 'value': row['ID']} 
                 for _, row in matches_df.iterrows()],
        placeholder="Select a Match",
        className="mb-3"
    ),
    html.Div(id='match-summary', className="mb-4"),
    
    # Player Performance Section
    html.H2("Player Performance Analysis", className="text-center text-danger"),
    dcc.Dropdown(
        id='player-dropdown',
        options=[],  # Populated dynamically based on match selection
        placeholder="Select a Player",
        className="mb-3"
    ),
    dbc.Row([
        dbc.Col(dcc.Graph(id='runs-trend'), md=4),
        dbc.Col(dcc.Graph(id='strike-rate-trend'), md=4),
        dbc.Col(dcc.Graph(id='economy-rate-trend'), md=4)
    ]),
    
    # Team Comparison Section
    html.H2("Team Comparison", className="text-center text-success"),
    dcc.Dropdown(
        id='team-dropdown',
        options=[{'label': team, 'value': team} for team in matches_df['Team1'].unique()],
        placeholder="Select a Team",
        className="mb-3"
    ),
    dcc.Graph(id='team-performance'),
], fluid=True)

# --- Callbacks ---

# Callback to display match summary and populate player dropdown
@app.callback(
    [Output('match-summary', 'children'),
     Output('player-dropdown', 'options')],
    [Input('match-dropdown', 'value')]
)
def update_match_summary(match_id):
    if not match_id:
        return "Select a match to view details.", []
    
    # Filter match data
    match_data = matches_df[matches_df['ID'] == match_id].iloc[0]
    
    # Match Summary Card
    summary = dbc.Card([
        dbc.CardBody([
            html.H4(f"{match_data['Team1']} vs {match_data['Team2']}", className="card-title text-success"),
            html.P(f"Venue: {match_data['Venue']}", className="card-text"),
            html.P(f"Date: {match_data['Date'].strftime('%Y-%m-%d')}", className="card-text"),
            html.P(f"Toss Winner: {match_data['TossWinner']} ({match_data['TossDecision']})", className="card-text"),
            html.P(f"Winner: {match_data['WinningTeam']} ({match_data['Margin']} {match_data['WonBy']})", 
                   className="card-text"),
            html.P(f"Player of the Match: {match_data['Player_of_Match']}", className="card-text")
        ])
    ], className="mb-3 shadow")
    
    # Populate player dropdown based on selected match
    players = eval(match_data['Team1Players']) + eval(match_data['Team2Players'])
    player_options = [{'label': player, 'value': player} for player in players]
    
    return summary, player_options

# Callback to update player performance charts
@app.callback(
    [Output('runs-trend', 'figure'), 
     Output('strike-rate-trend', 'figure'), 
     Output('economy-rate-trend', 'figure')],
    [Input('player-dropdown', 'value'),
     Input('match-dropdown', 'value')]
)
def update_player_performance(player_name, match_id):
    if not player_name or not match_id:
        return go.Figure(), go.Figure(), go.Figure()
    
    # Filter ball-by-ball data for the selected player and match
    player_data = ball_by_ball_df[(ball_by_ball_df['batter'] == player_name) & (ball_by_ball_df['ID'] == match_id)]
    
    if player_data.empty:
        return go.Figure(), go.Figure(), go.Figure()
    
    # Runs Trend Graph (Overwise)
    runs_trend = player_data.groupby(['overs'])['batsman_run'].sum().reset_index()
    runs_fig = px.line(runs_trend, x='overs', y='batsman_run', title=f'Runs Trend - {player_name}',
                       markers=True)
    
    # Strike Rate Trend (Overwise)
    strike_rate_trend = player_data.groupby(['overs'])[['batsman_run']].sum().reset_index()
    strike_rate_trend['balls_faced'] = player_data.groupby(['overs']).size().values  # Balls faced per over
    strike_rate_trend['strike_rate'] = (strike_rate_trend['batsman_run'] / strike_rate_trend['balls_faced']) * 100
    
    strike_rate_fig = px.line(strike_rate_trend, x='overs', y='strike_rate', title=f'Strike Rate Trend - {player_name}',
                              markers=True)
    
    # Economy Rate Trend (for bowlers)
    bowler_data = ball_by_ball_df[(ball_by_ball_df['bowler'] == player_name) & (ball_by_ball_df['ID'] == match_id)]
    
    if bowler_data.empty:
        economy_fig = go.Figure()
        
    else:
        economy_trend = bowler_data.groupby(['overs'])[['total_run']].sum().reset_index()
        economy_trend['balls_bowled'] = bowler_data.groupby(['overs']).size().values  # Balls bowled per over
        economy_trend['economy_rate'] = (economy_trend['total_run'] / economy_trend['balls_bowled']) * 6
        
        economy_fig = px.bar(economy_trend, x='overs', y='economy_rate', title=f'Economy Rate - {player_name}',
                             color_discrete_sequence=['#FF5733'])
    
    return runs_fig, strike_rate_fig, economy_fig

# Callback to update team performance chart
@app.callback(
    Output('team-performance', 'figure'),
    [Input('team-dropdown', 'value')]
)
def update_team_performance(selected_team):
    if not selected_team:
        return go.Figure()
    
    # Filter matches where the selected team played
    team_matches = matches_df[(matches_df['Team1'] == selected_team) | (matches_df['Team2'] == selected_team)]
    
    # Wins and Losses Count
    wins_losses = team_matches.groupby('WinningTeam').size().reset_index(name='count')
    
    fig = px.bar(wins_losses, x='WinningTeam', y='count', title=f'{selected_team} Performance',
                 color_discrete_sequence=['#3498DB'])
    
    return fig

# --- Run the App ---
if __name__ == "__main__":
    app.run(debug=True)
