
import re
import json
import pandas as pd
import numpy as np

html_path = "match_data.html"
with open(html_path, "r", encoding="utf-8") as f:
    html = f.read()

# Improved regex to find the JSON object
# It starts after require.config.params["args"] = 
# and ends with ; (or maybe the end of the script tag)
match = re.search(r'require\.config\.params\["args"\]\s*=\s*(\{.*?\});', html, re.DOTALL)
if not match:
    print("Regex failed to find require.config.params")
    # Fallback: look for matchCentreData
    match = re.search(r'matchCentreData\s*:\s*(\{.*?\})\s*\}', html, re.DOTALL)

if match:
    data_txt = match.group(1)
    print("Found data_txt snippet:", data_txt[:100])
    
    # Try to clean it up to be valid JSON
    # This is tricky because keys are not quoted.
    # We can use a simple regex to quote keys that are followed by :
    # But some values are also not quoted (like numbers).
    
    def quote_keys(json_str):
        # Quote keys: finding words followed by :
        # Avoid quoting inside already quoted strings if possible
        # A simple way is to use re.sub but it's risky.
        # Let's try to focus on the main keys we need.
        keys = ["matchId", "matchCentreData", "matchCentreEventTypeJson", "formationIdNameMappings", 
                "playerIdNameDictionary", "events", "home", "away", "players", "teamId", "name", 
                "value", "displayName", "minute", "second", "x", "y", "endX", "endY", "period", "type", "outcomeType"]
        for k in keys:
            json_str = json_str.replace(f"{k}:", f'"{k}":')
        return json_str

    # Actually, if we use a more general regex for keys:
    data_txt_quoted = re.sub(r'(\w+):', r'"\1":', data_txt)
    
    try:
        data = json.loads(data_txt_quoted)
        print("Success! JSON loaded.")
        events = data["matchCentreData"]["events"]
        df = pd.DataFrame(events)
        print("Events head:\n", df.head())
        print("Period type:", df['period'].dtype)
        print("Sample period:", df['period'].iloc[0])
        
        # Test period processing
        if isinstance(df['period'].iloc[0], dict):
            df['period_val'] = df['period'].apply(lambda x: x['value'])
            print("Max period:", df['period_val'].max())
        
    except Exception as e:
        print(f"JSON load failed: {e}")
else:
    print("Could not find match data.")
