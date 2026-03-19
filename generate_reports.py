import os
import json
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba
from matplotlib.gridspec import GridSpec
from mplsoccer import Pitch, VerticalPitch
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patheffects as path_effects
from unidecode import unidecode
import matplotlib.patches as patches

# Copy necessary functions from app.py logic
# (Simulated for this script)
from app import process_match, generate_all_reports

html_path = r"C:\Users\JOSAMU\Downloads\Real Madrid 3-0 Manchester City - Champions League 2025_2026 Live.html"
print(f"Processing match: {html_path}")
df, teams, players = process_match(html_path)
print("Generating all reports...")
images = generate_all_reports(df, teams)
print(f"Generated {len(images)} images.")
print("Done.")
