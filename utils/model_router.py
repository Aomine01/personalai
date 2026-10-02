import re

def should_use_deep_model(message_text: str) -> bool:
    """
    Determine if a message warrants routing to the deep fallback model (gpt-oss:20b).
    Uses simple, cheap, rule-based heuristics.
    """
    if not message_text:
        return False
    
    # 1. Message length threshold
    if len(message_text) > 400:
        return True
        
    # 2. Presence of code fences
    if "```" in message_text:
        return True
        
    # 3. Code/debugging keywords
    code_keywords = {
        "debug", "refactor", "stack trace", "stacktrace", "traceback",
        "pointer", "memory leak", "exception", "compile error",
        "syntax error", "nullpointer", "segfault", "recursion",
        "algorithm", "regex", "regular expression"
    }
    
    # Check lowercase words
    words = set(re.findall(r'\b[a-zA-Z0-9_\-+]{3,}\b', message_text.lower()))
    if words.intersection(code_keywords):
        return True
        
    # 4. Multi-part / numbered lists or multi-step requests
    multi_step_keywords = {"step-by-step", "step by step", "multi-step", "multi step"}
    for kw in multi_step_keywords:
        if kw in message_text.lower():
            return True
            
    # Check if there is a numbered list pattern like "1. ", "2. ", "3. "
    if re.search(r'\b\d+\.\s', message_text):
        return True
        
    return False
