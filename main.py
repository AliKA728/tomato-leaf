import os
import base64
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

# Load environment variables
load_dotenv()

# Initialize OpenAI client
MODEL = 'GPT-4.1'

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("CHRISKEY")
)

# Initialize FastAPI app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_methods=['*'],
    allow_credentials=True,
    allow_headers=['*'],
    allow_origins=['*'],
)

class ImageResponse(BaseModel):
    suggestions: str

class ChatbotResponse(BaseModel):
    response: str

class WeatherInsightsResponse(BaseModel):
    insights: str

def encode_image(image_file):
    return base64.b64encode(image_file).decode("utf-8")
    
def generate_suggestions(image_base64):
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system", 
                "content": "You are an expert tomato plant pathologist. Analyze the tomato leaf/plant image and provide: "
                          "1. Disease identification (if any) "
                          "2. Common symptoms "
                          "3. Treatment recommendations "
                          "4. Prevention strategies "
                          "Focus only on common tomato diseases like Early Blight, Late Blight, Septoria Leaf Spot, "
                          "Tomato Yellow Leaf Curl Virus, Blossom End Rot, etc."
            },
            {"role": "user", "content": [
                {"type": "text", "text": "Analyze this tomato leaf/plant for diseases and health:"},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{image_base64}"}
                }
            ]}
        ],
        temperature=0.0,
    )
    return response.choices[0].message.content

@app.post("/analyze-tomato-leaf", response_model=ImageResponse)
async def analyze_tomato_leaf(file: UploadFile = File(...)):
    image_data = await file.read()
    image_base64 = encode_image(image_data)
    suggestions = generate_suggestions(image_base64)
    return JSONResponse(content={"suggestions": suggestions})

def generate_chatbot_response(user_input):
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system", 
                "content": "You are a tomato cultivation expert. Provide advice only about: "
                          "- Tomato plant diseases "
                          "- Tomato varieties "
                          "- Soil requirements "
                          "- Watering needs "
                          "- Pest control "
                          "- Common cultivation problems "
                          "If asked about other crops, politely decline."
            },
            {"role": "user", "content": user_input}
        ],
        temperature=0.0,
    )
    return response.choices[0].message.content

@app.post("/tomato-chatbot", response_model=ChatbotResponse)
async def tomato_chatbot(query: str = Form(...)):
    chatbot_response = generate_chatbot_response(query)
    return JSONResponse(content={"response": chatbot_response})

def generate_tomato_weather_recommendation(temperature, humidity, windspeed, pressure):
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system", 
                "content": "You are an expert in tomato cultivation. Provide weather-specific recommendations for: "
                          "- Ideal planting times "
                          "- Disease prevention based on weather "
                          "- Watering adjustments "
                          "- Protection from extreme conditions "
                          "Focus only on tomato plants."
            },
            {"role": "user", "content": f"Current weather: Temp={temperature}°C, Humidity={humidity}%, "
                                      f"Wind={windspeed} m/s, Pressure={pressure} hPa. "
                                      f"What tomato-specific advice would you give?"}
        ],
        temperature=0.0,
    )
    return response.choices[0].message.content

@app.post("/tomato-weather-recommendation", response_model=WeatherInsightsResponse)
async def tomato_weather_recommendation(
    temperature: float = Form(...),
    humidity: float = Form(...),
    windspeed: float = Form(...),
    pressure: float = Form(...)
):
    insights = generate_tomato_weather_recommendation(temperature, humidity, windspeed, pressure)
    return JSONResponse(content={"insights": insights})