import json
import re
import os
import sys
import time  # 時間計測用に追加
import operator
import datetime  # 日付時刻の処理に必要
import base64  # base64エンコードされた画像の処理に必要
import asyncio
from typing_extensions import TypedDict
from typing import Any, Callable, Dict, Iterable, List, Optional, TypedDict, Annotated
from mcp.types import ImageContent

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI  # OpenAIのインポートを追加
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, SystemMessage, AnyMessage

from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

# 環境変数の設定
os.environ["LANGSMITH_TRACING"]="true"
os.environ["LANGSMITH_ENDPOINT"]="https://api.smith.langchain.com"
os.environ["LANGSMITH_API_KEY"]=os.getenv("LANGSMITH_API_KEY")
os.environ["LANGSMITH_PROJECT"]="newagentrag"
os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")

google_api_key = os.getenv("GOOGLE_APIKEY")
openai_api_key = os.getenv("OPENAI_API_KEY")  # OpenAIのAPIキーを環境変数から取得

class GraphState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    # system_message: Optional[SystemMessage]
    human_message: Optional[HumanMessage]

def create_graph(state: GraphState, tools, model_chain):
    def should_continue(state):
        messages = state["messages"]
        last_message = messages[-1]
        if last_message.tool_calls:
            return "tools"
        return END

    def call_model(state):
        messages = state["messages"]
        # 直前はtoolsのメッセージであるため、最後のメッセージを取得し、画像を保存する。
        last_message = messages[-1]

        # model_chain.invokeの実行時間を計測
        start_time = time.time()
        response = model_chain.invoke(messages)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"model_chain.invokeの実行時間: {execution_time:.2f}秒")

        return {"messages": [response]}


    tool_node = ToolNode(tools)
    
    workflow = StateGraph(state)
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", tool_node)

    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", should_continue, ["tools", END])
    workflow.add_edge("tools", "agent")
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    
    return app

def main(graph_config = {"configurable": {"thread_id": "12345"}}, query = None):
    # モデル設定の読み込み
    with open("config.json", "r") as f:
        config = json.load(f)
    
    # モデル設定の取得
    model_config = config.get("modelConfig", {})
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
    else:
        raise ValueError(f"未対応のプロバイダー: {provider}")

    # ツールの初期化
    tools = []

    # messageを作成する
    message = [
        SystemMessage(content= """
あなたは役にたつAIアシスタントです。日本語で回答し、考えた過程を結論より前に出力してください。
"""),
        MessagesPlaceholder("messages"),
    ]

    # messageからプロンプトを作成
    prompt = ChatPromptTemplate.from_messages(message)

    model_with_tools = prompt | model.bind(tools=tools)
    if query is None:
        # ユーザからの入力を取得
        query = input("入力してください:exitで終了: ")

    if query.lower() in ["exit", "quit"]:
        print("終了します。")
        return

    input_query = HumanMessage(
            [
                {
                    "type": "text",
                    "text": f"{query}"
                },
            ]
        )
    tools = []
    initial_state = {
        "messages": [input_query],
        # "system_message": query,
        "human_message": input_query
    }

    graph = create_graph(
        GraphState,
        tools,
        model_with_tools
    )

    response = graph.invoke(initial_state, graph_config)

    # デバック用
    # print("response: ", response)

    # 最終的な回答
    print("=================================")
    print(response["messages"][-1].content)


if __name__ == "__main__":
    
    main()
