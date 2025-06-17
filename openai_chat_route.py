import os
import openai
from fastapi import APIRouter, HTTPException, Body

router = APIRouter()
openai.api_key = os.getenv("OPENAI_API_KEY")

@router.post("/openai-chat")
async def openai_chat(data: dict = Body(...)):
    """Chat with OpenAI GPT-4.1 in Tatar language"""
    message = data.get("message")
    if not openai.api_key:
        raise HTTPException(status_code=500, detail="OpenAI API key not set")
    if not message:
        raise HTTPException(status_code=400, detail="Missing 'message' field")
    try:
        response = openai.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": "Син татарча сөйләшә торган ярдәмче. Сөйләм гади, аңлаешлы, һәм дусларча булсын."},
                {"role": "user", "content": message},
            ],
            temperature=0.8,
        )
        content = response.choices[0].message.content if response.choices else "что-то пошло не так"
        return {"response": content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI API error: {str(e)}")

