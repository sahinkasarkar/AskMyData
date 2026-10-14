import os
import sys
from dotenv import load_dotenv

load_dotenv()

print("=" * 45)
print("AskMyData: Environment Check")
print("=" * 45)

try:
    import pandas as pd
    import numpy as np
    import plotly
    import openpyxl
    import streamlit
    print("[OK] Core libraries ready (Pandas, Plotly, Streamlit, OpenPyXL)")
except ImportError as e:
    print(f"[FAIL] Missing dependency: {e}")
    sys.exit(1)

api_key = os.getenv("GEMINI_API_KEY")
if not api_key or api_key == "your_gemini_api_key_here":
    print("[WARN] GEMINI_API_KEY not set yet in .env (add it before AI testing)")
else:
    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        res = client.models.generate_content(
            model="gemini-3.8-flash",
            contents="Say 'System ready' in 2 words."
        )
        print(f"[OK] Gemini API Connected: {res.text.strip()}")
    except Exception as e:
        print(f"[FAIL] Gemini API check failed: {e}")

print("=" * 45)
print("Ready for Phase 2!")