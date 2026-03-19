
from app import extract_json_from_html, process_match, generate_all_reports
import os

html_path = r'C:\Users\JOSAMU\Downloads\Manchester City 1-2 Real Madrid - Champions League 2025_2026 Live.html'

print("Starting full process test...")
try:
    df, teams, players = process_match(html_path)
    print(f"Match processed successfully. Events: {len(df)}")
    print(f"Teams: {teams}")
    
    print("Generating reports...")
    images = generate_all_reports(df, teams)
    print("Reports generated successfully.")
    print(f"Generated images: {images}")
    
    # Check if files exist
    static_files = os.listdir('static')
    print(f"Files in static: {static_files}")
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"Error during full process: {e}")
