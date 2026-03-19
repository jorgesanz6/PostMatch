import os
import json
import re
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import requests
import cloudscraper
import chompjs
import threading
import time
import traceback
from bs4 import BeautifulSoup
from io import BytesIO
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
from mplsoccer import Pitch, VerticalPitch
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patheffects as path_effects
from flask import Flask, render_template, url_for, request, jsonify, send_from_directory
from unidecode import unidecode
import matplotlib.patches as patches
from scipy.spatial import ConvexHull
from PIL import Image
from urllib.request import urlopen
import base64

# Configuración de rutas
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Vercel handling: Use /tmp for generated images as the deployment is read-only
if os.environ.get('VERCEL'):
    STATIC_DIR = '/tmp'
else:
    STATIC_DIR = os.path.join(BASE_DIR, 'static')

XT_GRID_PATH = os.path.join(BASE_DIR, 'xT_Grid.csv')

if not os.path.exists(STATIC_DIR):
    os.makedirs(STATIC_DIR, exist_ok=True)

app = Flask(__name__, static_folder=STATIC_DIR)

# Colores personalizados (del script original)
green = '#69f900'
red = '#ff4b44'
blue = '#00a0de'
violet = '#a369ff'
bg_color= '#f5f5f5'
line_color= '#000000'
col1 = '#ff4b44'
col2 = '#00a0de'

# Cargar xT Grid al inicio
if os.path.exists(XT_GRID_PATH):
    try:
        xT_grid = pd.read_csv(XT_GRID_PATH, header=None)
        xT_grid = np.array(xT_grid)
    except:
        xT_grid = np.zeros((8, 12))
else:
    xT_grid = np.zeros((8, 12))

def extract_json_from_html(html_path):
    with open(html_path, 'r', encoding='utf-8', errors='ignore') as html_file:
        html = html_file.read()
    
    # 1. Intentar encontrar el objeto completo de argumentos
    match = re.search(r'require\.config\.params\["args"\]\s*=\s*(\{.*?\});', html, re.DOTALL)
    if match:
        data_txt = match.group(1)
    else:
        # 2. Si falla, buscar directamente matchCentreData
        match = re.search(r'matchCentreData\s*:\s*(\{.*?\})\s*,\s*matchId', html, re.DOTALL)
        if not match:
            # Tercer intento: buscar hasta el final del bloque de script o siguiente propiedad
            match = re.search(r'matchCentreData\s*:\s*(\{.*?)\n\s*\w+\s*:', html, re.DOTALL)
        
        if match:
            data_txt = match.group(0) # Incluimos la clave para envolverlo después si es necesario
            if not data_txt.startswith('{'):
                data_txt = '{' + data_txt + '}'
        else:
            return None

    # Limpieza de JSON no estándar (claves sin comillas, NaN, Infinity)
    # 1. Poner comillas a las claves que no las tienen: {key: -> {"key":
    data_txt = re.sub(r'([{,])\s*([a-zA-Z0-9_]+)\s*:', r'\1"\2":', data_txt)
    # 2. Reemplazar valores no válidos en JSON
    data_txt = data_txt.replace(':NaN', ':null').replace(':Infinity', ':null').replace(':-Infinity', ':null')
    
    # Intentar cargar
    try:
        data = json.loads(data_txt)
        # Si no tiene el nivel superior matchCentreData, envolverlo
        if "matchCentreData" not in data:
             # Si el objeto es el contenido de matchCentreData, envolverlo
             if "events" in data or "playerIdNameDictionary" in data:
                 return {"matchCentreData": data}
        return data
    except Exception as e:
        # Si falla por llaves extra al final (debido al dotall), intentar recorte agresivo
        try:
            last_brace = data_txt.rfind('}')
            if last_brace != -1:
                data = json.loads(data_txt[:last_brace+1])
                if "matchCentreData" not in data:
                    if "events" in data or "playerIdNameDictionary" in data:
                        return {"matchCentreData": data}
                return data
        except:
            pass
        print(f"Error parseando JSON: {e}")
        return None

def cumulative_match_mins(events_df):
    match_events = events_df.copy()
    period_map = {'FirstHalf': 1, 'SecondHalf': 2, 'FirstPeriodOfExtraTime': 3, 'SecondPeriodOfExtraTime': 4, 
                  'PenaltyShootout': 5, 'PostGame': 14, 'PreMatch': 16}
    
    if 'period' not in match_events.columns:
        return match_events

    if match_events['period'].dtype == object:
        # Extraer el 'value' si es un dict (ej: {'value': 2, 'displayName': 'SecondHalf'})
        def get_period_id(x):
            if isinstance(x, dict): return x.get('value', 1)
            # Intentar parsear si es un string que parece un dict
            if isinstance(x, str) and x.startswith('{'):
                try: 
                    import ast
                    d = ast.literal_eval(x)
                    if isinstance(d, dict): return d.get('value', 1)
                except: pass
            return x

        match_events['period_id'] = match_events['period'].apply(get_period_id).map(period_map).fillna(1)
        # Tambin obtener el displayName
        match_events['period'] = match_events['period'].apply(lambda x: x.get('displayName') if isinstance(x, dict) else x)
    else:
        match_events['period_id'] = match_events['period']

    match_events['cumulative_mins'] = match_events['minute'] + (1/60) * match_events['second'].fillna(0)
    
    # Asegurarse de que period_id sea int para np.arange
    match_events['period_id'] = pd.to_numeric(match_events['period_id'], errors='coerce').fillna(1).astype(int)
    
    for period in np.arange(1, int(match_events['period_id'].max()) + 1, 1):
        if period > 1:
            prev_period_events = match_events[match_events['period_id'] < period]
            curr_period_events = match_events[match_events['period_id'] == period]
            if not prev_period_events.empty and not curr_period_events.empty:
                t_delta = prev_period_events['cumulative_mins'].max() - curr_period_events['cumulative_mins'].min()
                match_events.loc[match_events['period_id'] == period, 'cumulative_mins'] += t_delta
    
    return match_events

def insert_ball_carries(events_df, min_carry_length=2, max_carry_length=60, min_carry_duration=1, max_carry_duration=12):
    match_events = events_df.reset_index(drop=True)
    match_carries = []
    
    for idx in range(len(match_events) - 1):
        match_event = match_events.iloc[idx]
        prev_evt_team = match_event['teamId']
        next_evt_idx = idx + 1
        
        next_evt = None
        temp_idx = next_evt_idx
        take_ons = 0
        while temp_idx < len(match_events):
            candidate = match_events.iloc[temp_idx]
            if candidate['type'] == 'TakeOn' and candidate['outcomeType'] == 'Successful':
                take_ons += 1
                temp_idx += 1
            elif ((candidate['type'] == 'TakeOn' and candidate['outcomeType'] == 'Unsuccessful')
                  or (candidate['teamId'] != prev_evt_team and candidate['type'] == 'Challenge' and candidate['outcomeType'] == 'Unsuccessful')
                  or (candidate['type'] == 'Foul')):
                temp_idx += 1
            else:
                next_evt = candidate
                break
        
        if next_evt is None: continue
        if 'endX' not in match_event or pd.isna(match_event['endX']): continue
        
        same_team = prev_evt_team == next_evt['teamId']
        not_ball_touch = match_event['type'] != 'BallTouch'
        dx = 105 * (match_event['endX'] - next_evt['x']) / 100
        dy = 68 * (match_event['endY'] - next_evt['y']) / 100
        dist_sq = dx**2 + dy**2
        far_enough = dist_sq >= min_carry_length**2
        not_too_far = dist_sq <= max_carry_length**2
        dt = 60 * (next_evt['cumulative_mins'] - match_event['cumulative_mins'])
        min_time = dt >= min_carry_duration
        same_phase = dt < max_carry_duration
        same_period = match_event['period'] == next_evt['period']
        
        if same_team and not_ball_touch and far_enough and not_too_far and min_time and same_phase and same_period:
            carry = match_event.copy()
            carry['type'] = 'Carry'
            carry['outcomeType'] = 'Successful'
            carry['x'] = match_event['endX']
            carry['y'] = match_event['endY']
            carry['endX'] = next_evt['x']
            carry['endY'] = next_evt['y']
            carry['cumulative_mins'] = (match_event['cumulative_mins'] + next_evt['cumulative_mins']) / 2
            carry['playerId'] = next_evt['playerId']
            carry['qualifiers'] = str([{'type': {'displayName': 'takeOns'}, 'value': str(take_ons)}])
            match_carries.append(carry)

    if match_carries:
        carries_df = pd.DataFrame(match_carries)
        match_events = pd.concat([match_events, carries_df], ignore_index=True)
        match_events = match_events.sort_values(['period_id', 'cumulative_mins']).reset_index(drop=True)
    
    return match_events

def add_xT(df):
    if xT_grid is None or xT_grid.sum() == 0:
        df['xT'] = 0.0
        return df
        
    xT_rows, xT_cols = xT_grid.shape
    df_xT = df.copy()
    df_xT['xT'] = 0.0
    
    mask = (df_xT['type_name'].isin(['Pass', 'Carry'])) & (df_xT['outcomeType_name'] == 'Successful')
    relevant = df_xT[mask].copy()
    if relevant.empty: return df_xT

    relevant['x1_bin'] = (relevant['x'] / 105 * xT_cols).astype(int).clip(0, xT_cols - 1)
    relevant['y1_bin'] = (relevant['y'] / 68 * xT_rows).astype(int).clip(0, xT_rows - 1)
    relevant['x2_bin'] = (relevant['endX'] / 105 * xT_cols).astype(int).clip(0, xT_cols - 1)
    relevant['y2_bin'] = (relevant['endY'] / 68 * xT_rows).astype(int).clip(0, xT_rows - 1)
    
    start_vals = relevant.apply(lambda r: xT_grid[r['y1_bin']][r['x1_bin']], axis=1)
    end_vals = relevant.apply(lambda r: xT_grid[r['y2_bin']][r['x2_bin']], axis=1)
    
    df_xT.loc[mask, 'xT'] = end_vals - start_vals
    return df_xT

def extract_data_from_dict(data):
    # WhoScored matchCentreData structure
    events_list = data["matchCentreData"]["events"]
    
    # Extraer nombres de equipos
    home_id = data["matchCentreData"]['home']['teamId']
    away_id = data["matchCentreData"]['away']['teamId']
    teams_dict = {
        home_id: data["matchCentreData"]['home']['name'],
        away_id: data["matchCentreData"]['away']['name']
    }
    
    # Extraer jugadores
    players_home = data["matchCentreData"]['home']['players']
    players_away = data["matchCentreData"]['away']['players']
    for p in players_home: p['teamId'] = home_id
    for p in players_away: p['teamId'] = away_id
    players_df = pd.concat([pd.DataFrame(players_home), pd.DataFrame(players_away)], ignore_index=True)
    
    return events_list, players_df, teams_dict

def process_advanced_data(df, teams_dict, players_df):
    # Limpieza de columnas de tipo dict (típico de WhoScored)
    for col in ['type', 'outcomeType', 'period']:
        if col in df.columns:
            df[f'{col}_name'] = df[col].apply(lambda x: x.get('displayName') if isinstance(x, dict) else x)
            df[col] = df[col].apply(lambda x: x.get('value') if isinstance(x, dict) else x)

    # Coordenadas y periodos
    df['endX'] = df['endX'].fillna(df['x'])
    df['endY'] = df['endY'].fillna(df['y'])
    
    # Procesamiento avanzado
    df = cumulative_match_mins(df)
    df = insert_ball_carries(df)
    df = add_xT(df)
    
    # Escalar a 105x68 (uefa scale)
    df['x'] *= 1.05
    df['y'] *= 0.68
    df['endX'] *= 1.05
    df['endY'] *= 0.68
    
    # Merge con info de jugadores
    df = df.merge(players_df[['playerId', 'name', 'shirtNo', 'position', 'isFirstEleven']], on='playerId', how='left')
    df['name'] = df['name'].apply(lambda x: unidecode(str(x)) if pd.notnull(x) else x)
    df['teamName'] = df['teamId'].map(teams_dict)
    
    # Métricas adicionales (prog_pass, prog_carry)
    df['dist_to_goal'] = np.sqrt((105 - df['x'])**2 + (34 - df['y'])**2)
    df['end_dist_to_goal'] = np.sqrt((105 - df['endX'])**2 + (34 - df['endY'])**2)
    df['prog_pass'] = np.where((df['type_name'] == 'Pass') & (df['outcomeType_name'] == 'Successful'),
                                df['dist_to_goal'] - df['end_dist_to_goal'], 0)
    df['prog_carry'] = np.where(df['type_name'] == 'Carry',
                                 df['dist_to_goal'] - df['end_dist_to_goal'], 0)
    
    return df

def process_match(html_path):
    data = extract_json_from_html(html_path)
    if not data:
        raise ValueError("No se pudo extraer la data del HTML proporcionado.")
        
    events_list, players_df, teams_dict = extract_data_from_dict(data)
    df = pd.DataFrame(events_list)
    df = process_advanced_data(df, teams_dict, players_df)
    
    return df, teams_dict, players_df

def get_passes_between_df(teamName, df, players_df):
    dfteam = df[(df['teamName'] == teamName) & (~df['type_name'].str.contains('SubstitutionOn|FormationChange|FormationSet|Card|SubstitutionOff', na=False))]
    
    # calculate median positions for player's passes
    average_locs_and_count_df = (dfteam.groupby('playerId').agg({'x': ['median'], 'y': ['median', 'count']}))
    average_locs_and_count_df.columns = ['pass_avg_x', 'pass_avg_y', 'count']
    average_locs_and_count_df = average_locs_and_count_df.merge(players_df[['playerId', 'name', 'shirtNo', 'position', 'isFirstEleven']], on='playerId', how='left')
    average_locs_and_count_df = average_locs_and_count_df.set_index('playerId')
    
    # calculate the number of passes between each position
    passes_df = dfteam[dfteam['type_name'] == 'Pass'].copy()
    passes_df['receiver'] = passes_df['playerId'].shift(-1) # Simplified for now, in a real scenario we'd need more logic
    
    passes_player_ids_df = passes_df.loc[:, ['playerId', 'receiver']].dropna()
    passes_player_ids_df['pos_max'] = (passes_player_ids_df[['playerId', 'receiver']].max(axis='columns'))
    passes_player_ids_df['pos_min'] = (passes_player_ids_df[['playerId', 'receiver']].min(axis='columns'))
    
    passes_between_df = passes_player_ids_df.groupby(['pos_min', 'pos_max']).size().reset_index(name='pass_count')
    
    # add on the location of each player
    passes_between_df = passes_between_df.merge(average_locs_and_count_df, left_on='pos_min', right_index=True)
    passes_between_df = passes_between_df.merge(average_locs_and_count_df, left_on='pos_max', right_index=True, suffixes=['', '_end'])
    
    return passes_between_df, average_locs_and_count_df

def plot_passing_network(df, teamName, ax, col):
    passes_between_df, average_locs_and_count_df = get_passes_between_df(teamName, df, df[['playerId', 'name', 'shirtNo', 'position', 'isFirstEleven']].drop_duplicates())
    
    MAX_LINE_WIDTH = 10
    passes_between_df['width'] = (passes_between_df.pass_count / passes_between_df.pass_count.max() * MAX_LINE_WIDTH)
    
    pitch = Pitch(pitch_type='uefa', pitch_color=bg_color, line_color=line_color, linewidth=2)
    pitch.draw(ax=ax)
    
    # Plotting lines
    pitch.lines(passes_between_df.pass_avg_x, passes_between_df.pass_avg_y, passes_between_df.pass_avg_x_end, passes_between_df.pass_avg_y_end,
                 lw=passes_between_df.width, color=col, zorder=1, alpha=0.5, ax=ax)
    
    # Plotting nodes
    for index, row in average_locs_and_count_df.iterrows():
        if row['isFirstEleven']:
            pitch.scatter(row['pass_avg_x'], row['pass_avg_y'], s=600, marker='o', color=bg_color, edgecolor=col, linewidth=2, alpha=1, ax=ax)
        else:
            pitch.scatter(row['pass_avg_x'], row['pass_avg_y'], s=600, marker='s', color=bg_color, edgecolor=col, linewidth=2, alpha=0.7, ax=ax)
        pitch.annotate(int(row['shirtNo']) if pd.notnull(row['shirtNo']) else '', xy=(row.pass_avg_x, row.pass_avg_y), c=col, ha='center', va='center', size=12, ax=ax)
        
    ax.set_title(f"Red de Pases: {teamName}", color=line_color, size=14, fontweight='bold')

def plot_momentum(df, hteam, ateam, ax):
    # Calcular momentum basado en xT acumulado por minuto
    momentum_df = df.copy()
    momentum_df['xT_val'] = momentum_df['xT']
    momentum_df.loc[momentum_df['teamName'] == ateam, 'xT_val'] *= -1
    
    mom = momentum_df.groupby('minute')['xT_val'].sum().reset_index()
    mom['average_xT'] = mom['xT_val'].rolling(window=3, min_periods=1).mean()
    
    colors = [col1 if x > 0 else col2 for x in mom['average_xT']]
    ax.bar(mom['minute'], mom['average_xT'], color=colors, alpha=0.8)
    
    ax.axhline(0, color=line_color, lw=1)
    ax.axvline(45, color='gray', linestyle='--', alpha=0.5)
    
    ax.set_title("Match Momentum (xT per Minute)", color=line_color, fontsize=16, fontweight='bold')
    ax.set_xlabel("Minute", color=line_color)
    ax.set_ylabel("Advantage", color=line_color)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    

def plot_shotmap(df, team_name, ax, color):
    pitch = Pitch(pitch_type='uefa', pitch_color=bg_color, line_color=line_color, linewidth=2)
    pitch.draw(ax=ax)
    
    shots = df[(df['teamName'] == team_name) & (df['type_name'].isin(['ShotOnPost', 'SavedShot', 'MissedShots', 'Goal']))]
    if shots.empty: return
    
    for _, shot in shots.iterrows():
        is_goal = shot['type_name'] == 'Goal'
        marker = 'o' if is_goal else 'x'
        size = 200 if is_goal else 100
        alpha = 1.0 if is_goal else 0.6
        ax.scatter(shot['x'], shot['y'], s=size, color=color, marker=marker, edgecolors=line_color if is_goal else None, zorder=3, alpha=alpha)
        
    ax.set_title(f"Mapa de Tiros: {team_name}", color=line_color, fontsize=14)

def plot_progressive_actions(df, team_name, ax, color, action_type='Pass'):
    pitch = Pitch(pitch_type='uefa', pitch_color=bg_color, line_color=line_color, linewidth=1)
    pitch.draw(ax=ax)
    
    col_name = 'prog_pass' if action_type == 'Pass' else 'prog_carry'
    type_filter = 'Pass' if action_type == 'Pass' else 'Carry'
    
    # WhoScored markers etc.
    actions = df[(df['teamName'] == team_name) & (df['type_name'] == type_filter) & (df[col_name] >= 7.0)] # 8+ meters
    
    if actions.empty: return
    
    for _, action in actions.iterrows():
        ax.annotate('', xy=(action['endX'], action['endY']), xytext=(action['x'], action['y']),
                    arrowprops=dict(arrowstyle='->', color=color, lw=1, alpha=0.4))
        
    ax.set_title(f"{'Pases' if action_type == 'Pass' else 'Conducciones'} Progresivos: {team_name}", color=line_color, fontsize=14)

def plot_box_entries(df, team_name, ax, color):
    pitch = Pitch(pitch_type='uefa', pitch_color=bg_color, line_color=line_color, linewidth=1)
    pitch.draw(ax=ax)
    
    # Entrada al área: termina en el área rival pero empieza fuera
    # Área rival WhoScored (uefa scale 105x68): x > 88.5, 13.6 < y < 54.4
    box_x = 88.5
    box_y_min = 13.6
    box_y_max = 54.4
    
    entries = df[(df['teamName'] == team_name) & 
                (df['endX'] >= box_x) & (df['endY'] >= box_y_min) & (df['endY'] <= box_y_max) & 
                ~((df['x'] >= box_x) & (df['y'] >= box_y_min) & (df['y'] <= box_y_max)) &
                (df['type_name'].isin(['Pass', 'Carry'])) & (df['outcomeType_name'] == 'Successful')]
    
    if entries.empty: return
    
    for _, entry in entries.iterrows():
        ax.annotate('', xy=(entry['endX'], entry['endY']), xytext=(entry['x'], entry['y']),
                    arrowprops=dict(arrowstyle='->', color=color, lw=1.5, alpha=0.6))
        
    ax.set_title(f"Entradas al Área: {team_name}", color=line_color, fontsize=14)

def plot_team_defensive_actions(df, team_name, ax, color):
    pitch = Pitch(pitch_type='uefa', pitch_color=bg_color, line_color=line_color, linewidth=2)
    pitch.draw(ax=ax)
    
    def_types = ['Tackle', 'Interception', 'BallRecovery', 'Clearance', 'BlockedPass']
    def_acts = df[(df['teamName'] == team_name) & (df['type_name'].isin(def_types))]
    
    if def_acts.empty: return
    
    # Heatmap of defensive actions
    sns.kdeplot(x=def_acts.x, y=def_acts.y, ax=ax, fill=True, cmap='Blues', alpha=0.3, levels=5)
    
    # Significant actions
    for _, act in def_acts.iterrows():
        ax.scatter(act['x'], act['y'], s=80, color=color, marker='o', edgecolors='black', zorder=3, alpha=0.6)
        
    ax.set_title(f"Acciones Defensivas Totales: {team_name}", color=line_color, fontsize=14, fontweight='bold')

def plot_player_heatmap(df, pname, ax):
    pitch = Pitch(pitch_type='uefa', pitch_color=bg_color, line_color=line_color, linewidth=2)
    pitch.draw(ax=ax)
    
    player_df = df[df['name'] == pname]
    if player_df.empty: return
    
    # Simple heatmap using touches
    touches = player_df[player_df['isTouch'] == True]
    if not touches.empty:
        sns.kdeplot(x=touches.x, y=touches.y, ax=ax, fill=True, cmap='Reds', alpha=0.5, levels=10)
    
    ax.set_title(f"Mapa de Calor: {pname}", color=line_color, fontsize=12)

def plot_player_passes(df, pname, ax, color):
    pitch = Pitch(pitch_type='uefa', pitch_color=bg_color, line_color=line_color, linewidth=2)
    pitch.draw(ax=ax)
    
    passes = df[(df['name'] == pname) & (df['type_name'] == 'Pass')]
    if passes.empty: return
    
    for _, p in passes.iterrows():
        alpha = 0.6 if p['outcomeType_name'] == 'Successful' else 0.2
        ax.annotate('', xy=(p['endX'], p['endY']), xytext=(p['x'], p['y']),
                    arrowprops=dict(arrowstyle='->', color=color if p['outcomeType_name'] == 'Successful' else 'gray', lw=1, alpha=alpha))
    
    ax.set_title(f"Pases: {pname}", color=line_color, fontsize=12)

def plot_player_defensive_acts(df, pname, ax, color):
    pitch = Pitch(pitch_type='uefa', pitch_color=bg_color, line_color=line_color, linewidth=2)
    pitch.draw(ax=ax)
    
    def_types = ['Tackle', 'Interception', 'BallRecovery', 'Clearance', 'BlockedPass', 'Foul']
    def_acts = df[(df['name'] == pname) & (df['type_name'].isin(def_types))]
    
    if def_acts.empty: return
    
    for _, act in def_acts.iterrows():
        ax.scatter(act['x'], act['y'], s=150, color=color, marker='X' if act['type_name'] == 'Foul' else 'o', edgecolors='black', zorder=3, alpha=0.8)
        
    ax.set_title(f"Acciones Defensivas: {pname}", color=line_color, fontsize=12, fontweight='bold')

def generate_player_dashboard(df, pname, team_name, color):
    # Calcular estadísticas resumidas
    player_df = df[df['name'] == pname]
    passes = player_df[player_df['type_name'] == 'Pass']
    acc_passes = passes[passes['outcomeType_name'] == 'Successful']
    pass_acc = (len(acc_passes) / len(passes) * 100) if not passes.empty else 0
    total_xt = player_df['xT'].sum()
    prog_dist = player_df['prog_pass'].sum() + player_df['prog_carry'].sum()
    def_acts = player_df[player_df['type_name'].isin(['Tackle', 'Interception', 'BallRecovery', 'Clearance', 'BlockedPass', 'Foul'])]
    
    # Crear dashboard 3x2 para más información
    fig = plt.figure(figsize=(18, 12), facecolor=bg_color)
    gs = GridSpec(2, 3, figure=fig)
    
    # 1. Mapa de Calor (0, 0)
    ax1 = fig.add_subplot(gs[0, 0])
    plot_player_heatmap(df, pname, ax1)
    
    # 2. Pases (0, 1)
    ax2 = fig.add_subplot(gs[0, 1])
    plot_player_passes(df, pname, ax2, color)
    
    # 3. Acciones Defensivas (0, 2)
    ax3 = fig.add_subplot(gs[0, 2])
    plot_player_defensive_acts(df, pname, ax3, color)
    
    # 4. Conducciones (1, 0)
    ax4 = fig.add_subplot(gs[1, 0])
    pitch = Pitch(pitch_type='uefa', pitch_color=bg_color, line_color=line_color, linewidth=2)
    pitch.draw(ax=ax4)
    carries = player_df[player_df['type'] == 'Carry']
    for _, c in carries.iterrows():
        ax4.annotate('', xy=(c['endX'], c['endY']), xytext=(c['x'], c['y']),
                          arrowprops=dict(arrowstyle='->', color=color, lw=1.5, alpha=0.6))
    ax4.set_title(f"Conducciones: {pname}", color=line_color, fontsize=12, fontweight='bold')
    
    # 5. Tiros / xG (1, 1)
    ax5 = fig.add_subplot(gs[1, 1])
    pitch.draw(ax=ax5)
    shots = player_df[player_df['type_name'].isin(['ShotOnPost', 'SavedShot', 'MissedShots', 'Goal'])]
    for _, shot in shots.iterrows():
        is_goal = shot['type_name'] == 'Goal'
        ax5.scatter(shot['x'], shot['y'], s=200 if is_goal else 100, color='gold' if is_goal else color, 
                   marker='o' if is_goal else 'x', edgecolors='black', zorder=3)
    ax5.set_title(f"Tiros: {pname}", color=line_color, fontsize=12, fontweight='bold')
    
    # 6. Panel de Estadísticas (1, 2)
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    stats_text = (
        f"{pname.upper()}\n"
        f"{team_name}\n\n"
        f"Pases Totales: {len(passes)}\n"
        f"Precisión: {pass_acc:.1f}%\n"
        f"xT Generado: {total_xt:.3f}\n"
        f"Progreso Balón: {prog_dist:.1f}m\n"
        f"Acc. Defensivas: {len(def_acts)}\n"
    )
    ax6.text(0.5, 0.5, stats_text, ha='center', va='center', fontsize=20, 
            color=line_color, fontweight='bold', bbox=dict(facecolor='white', alpha=0.1, pad=10))
    
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.savefig(os.path.join(STATIC_DIR, f"player_{pname.replace(' ', '_')}.png"))
    plt.close()
    
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    return f"data:image/png;base64,{img_base64}"

def generate_global_summary(df, players_df, teams_dict):
    team_names = list(teams_dict.values())
    if not team_names: return "Resumen no disponible."
    hteam, ateam = team_names[0], team_names[1]
    
    try:
        # 1. MVP (highest xT)
        xt_sums = df.groupby('name')['xT'].sum()
        top_xt_idx = xt_sums.idxmax()
        top_xt_val = xt_sums.max()
        
        # 2. Top Shooter
        shots_df = df[df['type_name'].isin(['ShotOnPost', 'SavedShot', 'MissedShots', 'Goal'])]
        top_shooter_idx = shots_df.groupby('name').size().idxmax() if not shots_df.empty else "N/A"
        
        # 3. Top Defensive
        def_types = ['Tackle', 'Interception', 'BallRecovery', 'Clearance', 'BlockedPass']
        def_df = df[df['type_name'].isin(def_types)]
        top_def_idx = def_df.groupby('name').size().idxmax() if not def_df.empty else "N/A"
        
        # 4. Success rates
        summary = f"### Análisis Postpartido: {hteam} vs {ateam}\n\n"
        summary += f"El encuentro entre **{hteam}** y **{ateam}** dejó grandes actuaciones individuales. "
        summary += f"**{top_xt_idx}** fue el motor ofensivo más peligroso, generando un **xT acumulado de {top_xt_val:.3f}**, "
        summary += "lo que indica una gran capacidad para mover el balón a zonas de finalización.\n\n"
        
        if top_shooter_idx != "N/A":
            summary += f"- **Amenaza en Área**: **{top_shooter_idx}** lideró los intentos de cara a puerta, siendo el jugador con más tiros totales.\n"
        if top_def_idx != "N/A":
            summary += f"- **Seguridad Defensiva**: **{top_def_idx}** fue fundamental en la fase de no posesión, destacando en recuperaciones y acciones defensivas exitosas.\n"
        
        summary += "\nEste análisis combina métricas avanzadas de posicionamiento, creación de peligro (xT) y volumen de acciones para determinar el impacto real de cada jugador en el resultado."
        return summary
    except Exception as e:
        return f"Error generando resumen: {str(e)}"

def save_granular_chart(df, team_name, plot_func, filename, *args):
    fig, ax = plt.subplots(figsize=(12, 8), facecolor=bg_color)
    plot_func(df, team_name, ax, *args)
    plt.tight_layout()
    plt.savefig(os.path.join(STATIC_DIR, filename))
    
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    return f"data:image/png;base64,{img_base64}"

def generate_all_reports(df, teams_dict):
    team_names = list(teams_dict.values())
    hteam, ateam = team_names[0], team_names[1]
    
    results = {
        'hteam': hteam,
        'ateam': ateam,
        'categories': {
            'General': [],
            'Ataque': [],
            'Transición': [],
            'Defensa': [],
            'Jugadores': []
        }
    }
    
    # 1. Momentum (Global)
    fig, ax = plt.subplots(figsize=(16, 6), facecolor=bg_color)
    plot_momentum(df, hteam, ateam, ax)
    plt.tight_layout()
    plt.savefig(os.path.join(STATIC_DIR, 'match_momentum.png'))
    
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    mom_base64 = base64.b64encode(buf.read()).decode('utf-8')
    
    results['categories']['General'].append({'name': f'data:image/png;base64,{mom_base64}', 'label': 'Momentum del Partido'})
    
    # 2. Granular Team Charts
    # Home Team
    results['categories']['General'].append({
        'name': save_granular_chart(df, hteam, plot_passing_network, f'passing_network_{hteam}.png', col1),
        'label': f'Red de Pases: {hteam}', 'team': hteam
    })
    results['categories']['Ataque'].append({
        'name': save_granular_chart(df, hteam, plot_shotmap, f'shotmap_{hteam}.png', col1),
        'label': f'Mapa de Tiros: {hteam}', 'team': hteam
    })
    results['categories']['Transición'].append({
        'name': save_granular_chart(df, hteam, plot_progressive_actions, f'prog_passes_{hteam}.png', col1, 'Pass'),
        'label': f'Pases Progresivos: {hteam}', 'team': hteam
    })
    results['categories']['Transición'].append({
        'name': save_granular_chart(df, hteam, plot_progressive_actions, f'prog_carries_{hteam}.png', col1, 'Carry'),
        'label': f'Conducciones Progresivas: {hteam}', 'team': hteam
    })
    results['categories']['Ataque'].append({
        'name': save_granular_chart(df, hteam, plot_box_entries, f'box_entries_{hteam}.png', col1),
        'label': f'Entradas al Área: {hteam}', 'team': hteam
    })
    results['categories']['Defensa'].append({
        'name': save_granular_chart(df, hteam, plot_team_defensive_actions, f'def_actions_{hteam}.png', col1),
        'label': f'Mapa Defensivo: {hteam}', 'team': hteam
    })
    
    # Away Team
    results['categories']['General'].append({
        'name': save_granular_chart(df, ateam, plot_passing_network, f'passing_network_{ateam}.png', col2),
        'label': f'Red de Pases: {ateam}', 'team': ateam
    })
    results['categories']['Ataque'].append({
        'name': save_granular_chart(df, ateam, plot_shotmap, f'shotmap_{ateam}.png', col2),
        'label': f'Mapa de Tiros: {ateam}', 'team': ateam
    })
    results['categories']['Transición'].append({
        'name': save_granular_chart(df, ateam, plot_progressive_actions, f'prog_passes_{ateam}.png', col2, 'Pass'),
        'label': f'Pases Progresivos: {ateam}', 'team': ateam
    })
    results['categories']['Transición'].append({
        'name': save_granular_chart(df, ateam, plot_progressive_actions, f'prog_carries_{ateam}.png', col2, 'Carry'),
        'label': f'Conducciones Progresivas: {ateam}', 'team': ateam
    })
    results['categories']['Ataque'].append({
        'name': save_granular_chart(df, ateam, plot_box_entries, f'box_entries_{ateam}.png', col2),
        'label': f'Entradas al Área: {ateam}', 'team': ateam
    })
    results['categories']['Defensa'].append({
        'name': save_granular_chart(df, ateam, plot_team_defensive_actions, f'def_actions_{ateam}.png', col2),
        'label': f'Mapa Defensivo: {ateam}', 'team': ateam
    })
    
    # 3. Jugadores
    all_players = df.groupby(['name', 'teamName']).size().reset_index(name='count')
    all_players = all_players[all_players['count'] >= 5]
    
    for _, row in all_players.iterrows():
        pname = row['name']
        tname = row['teamName']
        if pname == 'Unknown' or not pname: continue
        
        color = col1 if tname == hteam else col2
        try:
            img_path = generate_player_dashboard(df, pname, tname, color)
            results['categories']['Jugadores'].append({
                'name': img_path, 
                'label': f'Dashboard: {pname}', 
                'team': tname,
                'player': pname
            })
        except Exception as e:
            print(f"Error generando dashboard para {pname}: {e}")
            
    return results

# Estado global del análisis
analysis_progress = {
    'status': 'Listo',
    'progress': 0
}

def run_analysis_task(url, match_id):
    global analysis_progress, match_data_cache
    print(f"DEBUG: Starting analysis task for URL: {url}")
    try:
        analysis_progress['status'] = 'Estableciendo conexin segura (Cloudscraper)...'
        analysis_progress['progress'] = 5
        
        scraper = cloudscraper.create_scraper()
        
        analysis_progress['status'] = 'Descargando datos del partido...'
        analysis_progress['progress'] = 15
        print("DEBUG: Fetching URL...")
        response = scraper.get(url, timeout=30)
        
        if response.status_code != 200:
            print(f"DEBUG: Fetch failed with status {response.status_code}")
            raise Exception(f"No se pudo acceder a la URL: {response.status_code}. Anti-bot block?")
        
        html_content = response.text
        print(f"DEBUG: Downloaded {len(html_content)} bytes")
        
        analysis_progress['status'] = 'Buscando matchCentreData en el HTML...'
        analysis_progress['progress'] = 25
        
        match = re.search(r'require\.config\.params\["args"\]\s*=\s*(\{.*?\});', html_content, re.DOTALL)
        if not match:
            print("DEBUG: Regex match failed for matchCentreData")
            raise Exception("No se encontr matchCentreData. Verifique que la URL sea un Live de WhoScored.")
            
        data = chompjs.parse_js_object(match.group(1))
        print("DEBUG: JSON parsed successfully (using chompjs.parse_js_object)")
        
        events_list, players_df, teams = extract_data_from_dict(data)
        print(f"DEBUG: Processing match data - Total events: {len(events_list)}")
        
        df = pd.DataFrame(events_list)
        
        # Mantener solo columnas esenciales
        essential_cols = ['x', 'y', 'endX', 'endY', 'teamId', 'playerId', 'type', 'outcomeType', 'period', 'minute', 'second', 'isTouch']
        available_cols = [c for c in essential_cols if c in df.columns]
        df = df[available_cols].copy()

        # Laundry: SOLO stringificar columnas que no son lógicas y pueden tener unhashables
        # Excluimos 'period', 'type', 'outcomeType' porque se procesan luegos
        for col in df.columns:
            if col not in ['period', 'type', 'outcomeType']:
                if df[col].apply(lambda x: isinstance(x, (list, dict))).any():
                    df[col] = df[col].astype(str)
        
        # Laundry para players_df
        for col in players_df.columns:
             if players_df[col].apply(lambda x: isinstance(x, (list, dict))).any():
                players_df[col] = players_df[col].astype(str)

        df = process_advanced_data(df, teams, players_df)
        
        analysis_progress['status'] = 'Limpiando reportes anteriores...'
        analysis_progress['progress'] = 55
        for f in os.listdir(STATIC_DIR):
            if f.endswith('.png'):
                try: os.remove(os.path.join(STATIC_DIR, f))
                except: pass
                
        analysis_progress['status'] = 'Generando Reportes Granulares...'
        analysis_progress['progress'] = 65
        reports_data = generate_all_reports(df, teams)
        
        analysis_progress['status'] = 'Generando Resumen del Partido...'
        analysis_progress['progress'] = 90
        summary = generate_global_summary(df, players_df, teams)
        
        # Actualizar caché
        match_data_cache['df'] = df
        match_data_cache['teams'] = teams
        match_data_cache['players'] = players_df
        match_data_cache['images'] = reports_data['categories']
        match_data_cache['summary'] = summary
        match_data_cache['hteam'] = reports_data['hteam']
        match_data_cache['ateam'] = reports_data['ateam']
        
        analysis_progress['status'] = 'Análisis Completado con éxito!'
        analysis_progress['progress'] = 100
        print("DEBUG: Analysis complete")
        
    except Exception as e:
        print(f"DEBUG: CRITICAL ERROR in run_analysis_task: {e}")
        error_log = os.path.join(STATIC_DIR, 'CRITICAL_ERROR.txt')
        with open(error_log, 'a') as f:
            f.write(f"\n--- ERROR {time.ctime()} ---\n")
            traceback.print_exc(file=f)
        analysis_progress['status'] = f"ERROR: {str(e)}"
        # No resetear a 0 para que el error se vea en la barra de progreso (roja?) 
        # o al menos se quede el mensaje.

@app.route('/analyze', methods=['POST'])
def analyze():
    global analysis_progress
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No JSON data received"}), 400
            
        url = data.get('url')
        match_id = data.get('match_id')
        
        debug_log = os.path.join(STATIC_DIR, 'server_debug.log')
        with open(debug_log, 'a') as f:
            f.write(f"{time.ctime()}: DEBUG: /analyze reached. URL: {url}\n")
        
        print(f"DEBUG: /analyze reached. URL: {url}")
        
        if not url:
            return jsonify({"error": "URL requerida"}), 400
            
        # Reset status
        analysis_progress['status'] = 'Iniciando...'
        analysis_progress['progress'] = 1
        
        # Iniciar tarea
        if os.environ.get('VERCEL'):
            # En Vercel, ejecutamos de forma síncrona para evitar que el proceso se mate
            run_analysis_task(url, match_id)
            return jsonify({
                "status": "Completado",
                "results": {
                    "images": match_data_cache.get('images'),
                    "summary": match_data_cache.get('summary'),
                    "hteam": match_data_cache.get('hteam'),
                    "ateam": match_data_cache.get('ateam')
                }
            })
        else:
            # En local, mantenemos el segundo plano
            thread = threading.Thread(target=run_analysis_task, args=(url, match_id))
            thread.daemon = True
            thread.start()
            return jsonify({"message": "Anlisis iniciado"})
    except Exception as e:
        print(f"DEBUG: Error in /analyze route: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/status')
def status():
    return jsonify(analysis_progress)

# Estado global
match_data_cache = {
    'df': None,
    'teams': None,
    'players': None,
    'images': {},
    'summary': '',
    'hteam': '',
    'ateam': ''
}

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory(STATIC_DIR, filename)

@app.route('/')
def index():
    # ... (rest of the code)
    global match_data_cache
    title = "Análisis Postpartido"
    images = match_data_cache.get('images', {})
    summary = match_data_cache.get('summary', '')
    hteam = match_data_cache.get('hteam', '')
    ateam = match_data_cache.get('ateam', '')
    
    return render_template('index.html', 
                         title=title, 
                         images=images, 
                         summary=summary,
                         hteam=hteam,
                         ateam=ateam)

if __name__ == '__main__':
    if not os.path.exists(STATIC_DIR):
        os.makedirs(STATIC_DIR)
    # Desactivamos debug para evitar duplicidad de procesos y problemas con hilos en Windows
    app.run(debug=False, port=55555, host='0.0.0.0')
