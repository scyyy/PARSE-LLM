import re
import os
import random
from langchain_core.tools import tool


@tool
def regular_expression_executor(system, regular_expression):
    """Remove log headers by regular expressions."""
    with open("../datasets"+system+"_full.log", "r", encoding="utf-8") as f:
        lines = f.readlines()
    logs = [i.strip() for i in lines]

    fail_count = 0
    fail_log = []
    cleaned_messages = [re.sub(regular_expression, '', message) for message in logs]
    for i in range(len(cleaned_messages)):
        if logs[i] == cleaned_messages[i]:
            fail_count += 1
            fail_log.append(logs[i])

    if fail_count == 0:
        return "The regular expression is perfect!"
    else:
        example = random.sample(fail_log, 5)
        return "Imperfect regular expressions, cannot be applied to all logs, for example:\n" + "\n".join(example)