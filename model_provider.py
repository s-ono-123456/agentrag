import json
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

# 環境変数からAPIキーを取得
google_api_key = os.getenv("GOOGLE_APIKEY")
openai_api_key = os.getenv("OPENAI_API_KEY")

def load_model_config():
    """configファイルからモデル設定を読み込む"""
    with open("config.json", "r") as f:
        config = json.load(f)
    return config.get("modelConfig", {})

def get_language_model():
    """モデル設定に基づいて適切な言語モデルを初期化して返す"""
    # モデル設定の読み込み
    model_config = load_model_config()
    provider = model_config.get("provider", "google")  # デフォルトはGoogle
    
    # モデルプロバイダーに基づいてモデルを初期化
    if provider == "google":
        if not google_api_key:
            raise ValueError("GOOGLE_APIKEYが設定されていません")
        
        google_model_config = model_config.get("models", {}).get("google", {})
        model = ChatGoogleGenerativeAI(
            model=google_model_config.get("model", "gemini-2.0-flash"),
            google_api_key=google_api_key,
            temperature=google_model_config.get("temperature", 0.1),
        )
        print("Google Geminiモデルを使用します")
        return model
    elif provider == "openai":
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEYが設定されていません")
        
        openai_model_config = model_config.get("models", {}).get("openai", {})
        model = ChatOpenAI(
            model=openai_model_config.get("model", "gpt-4.1-mini"),
            openai_api_key=openai_api_key,
            temperature=openai_model_config.get("temperature", 0.1),
        )
        print("OpenAIモデルを使用します")
        return model
    else:
        raise ValueError(f"未対応のプロバイダー: {provider}")