import re


def sanitize_prompt(prompt: str) -> str:
    """
    清理 prompt，移除 lora 标签和重复逗号
    """
    prompt_text = prompt or ""
    prompt_text = re.sub(r"<lora(.+?)>", "", prompt_text, flags=re.IGNORECASE)
    prompt_text = re.sub(r"(,)\1+", r"\1", prompt_text).strip()
    return prompt_text

