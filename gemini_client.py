import google.generativeai as genai

GEMINI_API_KEY = "YOUR API KEY"
genai.configure(api_key = GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.0-flash-lite")

for m in genai.list_models():
    if "embedding" in m.name:
        print(m.name, " → ", m.supported_generation_methods)