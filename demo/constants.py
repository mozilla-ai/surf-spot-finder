MODEL_OPTIONS = [
    # "huggingface/novita/deepseek-ai/DeepSeek-V3",
    # "huggingface/novita/meta-llama/Llama-3.3-70B-Instruct",
    "openai/gpt-4.1-nano",
    "openai/gpt-4.1-mini",
    "openai/gpt-4o",
    "gemini/gemini-2.0-flash-lite",
    "gemini-2.0-flash",
    # "huggingface/Qwen/Qwen3-32B", # right now throwing an internal error, but novita qwen isn't supporting tool calling
]

# Novita was the only HF based provider that worked.

# Hugginface API Provider Error:
# Must alternate between assistant/user, which meant that the 'tool' role made it puke
