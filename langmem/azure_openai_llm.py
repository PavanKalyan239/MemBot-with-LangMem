import os
import base64
import requests
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from logger_config import configure_logging

load_dotenv()
logger = configure_logging("logs", "application.log")

model = os.getenv("AZURE_MODEL")
azure_endpoint = os.getenv("AZURE_ENDPOINT")
api_version = os.getenv("API_VERSION")
azure_openai_api_key = os.getenv("AZURE_OPEN_AI_API_KEY")


api_key_url = os.getenv("API_KEY_URL")
client_id = os.getenv("CLIENT_ID")
client_secret = os.getenv("CLIENT_SECRET")
appkey = os.getenv("APPKEY")


def encode_credentials(client_id, client_secret):
    credentials = f"{client_id}:{client_secret}"
    return base64.b64encode(credentials.encode("utf-8")).decode("utf-8")


def get_token(encoded_value):
    url = api_key_url
    headers = {
        "Accept": "*/*",
        "Content-Type": "application/x-www-form-urlencoded",
        "Authorization": f"Basic {encoded_value}",
    }
    data = {"grant_type": "client_credentials"}
    response = requests.post(url, headers=headers, data=data)
    if response.status_code == 200:
        return response.json().get("access_token")
    else:
        return response.text


def get_llm():
    if azure_openai_api_key and azure_openai_api_key.strip():
        logger.info("Azure OpenAI API Key found")
        llm = AzureOpenAIEmbeddings(
            deployment_name=model,
            azure_endpoint=azure_endpoint,
            api_key=azure_openai_api_key,
            api_version=api_version,
        )
        return llm
    elif client_id and client_secret:
        logger.info("Client ID and Client Secret found")
        encoded_value = encode_credentials(client_id, client_secret)
        token = get_token(encoded_value)
        user = f'{{"appkey": "{appkey}"}}'

        llm = AzureChatOpenAI(
            deployment_name=model,
            azure_endpoint=azure_endpoint,
            api_key=token,
            api_version=api_version,  # Optional if specific base URL is needed
            model_kwargs=dict(user=user),  # Include API key if necessary
        )
        return llm
    else:
        logger.error("Azure OpenAI API Key or Client ID and Client Secret not found")
        return None
