import google.generativeai as genai

# Load your key manually here just to test
genai.configure(api_key="AIzaSyBnaH4KBt2tVymInOScD6FYnabUy216UME") 

print("Available models:")
for m in genai.list_models():
    if 'generateContent' in m.supported_generation_methods:
        print(f"- {m.name}")