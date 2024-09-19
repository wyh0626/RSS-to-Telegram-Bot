import openai
import json
from . import log
from src.env import OPENAI_API_KEY, OPENAI_API_BASE, AI_SUMMARY_ENABLED, AI_SUMMARY_MODEL, AI_SUMMARY_PROMPT

logger = log.getLogger('RSStT.openai_helper')

# 设置 OpenAI API 的基础 URL
openai.api_base = OPENAI_API_BASE

async def summarize_content(content: str) -> str:
    if not AI_SUMMARY_ENABLED:
        return content  # 如果AI总结功能未启用,直接返回原内容

    try:
        # 准备请求内容
        messages = [
            {"role": "system", "content": AI_SUMMARY_PROMPT},
            {"role": "user", "content": content}
        ]
        
        # 打印请求内容
        print("OpenAI 请求内容:")
        print(json.dumps({
            "model": AI_SUMMARY_MODEL,
            "messages": messages,
            "max_tokens": 4000
        }, indent=2, ensure_ascii=False))
        
        # 发送请求
        response = await openai.ChatCompletion.acreate(
            model=AI_SUMMARY_MODEL,
            messages=messages,
            max_tokens=4000,
            api_key=OPENAI_API_KEY
        )
        
        # 打印回复内容
        print("\nOpenAI 回复内容:")
        print(json.dumps(response, indent=2, ensure_ascii=False))
        
        return response.choices[0].message['content'].strip()
    except Exception as e:
        logger.error(f"OpenAI API调用失败: {e}")
        return content  # 如果API调用失败,返回原内容