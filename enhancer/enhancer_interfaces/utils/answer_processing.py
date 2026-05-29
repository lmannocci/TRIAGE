import ast
import numpy as np
import pandas as pd
import re
import json



def safe_parse_answer(answer: str) -> dict:
    """
    Extract and parse the first JSON object found in a MedRAG answer.
    """
    match = re.search(r'\{.*\}', answer, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in answer")

    json_str = match.group(0)
    return json.loads(json_str)



def parse_string(answer, lm):
    result = {}

    # 1. step_by_step_thinking
    match = re.search(r'"step_by_step_thinking"\s*:\s*"(.+?)"\s*"classification"', answer, re.DOTALL)
    if match:
        result['step_by_step_thinking'] = match.group(1)

    # 2. classification
    match = re.search(r'"classification"\s*:\s*(\d+)', answer)
    if match:
        result['classification'] = int(match.group(1))

    # 3. feature_importance_ranking
    match = re.search(r'"feature_importance_ranking"\s*:\s*(\[[^\]]*\])', answer)
    if match:
        # convert single quotes to double quotes for JSON parsing
        try:
            result['feature_importance_ranking'] = [x.strip().strip('"').strip("'") 
                                                for x in re.findall(r"'(.*?)'|\"(.*?)\"", match.group(1))]
        except Exception:
            result['feature_importance_ranking'] = None
    # 4. rules_applied using regex (no json.loads)
    rules_match = re.search(r'"rules_applied"\s*:\s*\{(.+?)\}\s*$', answer, re.DOTALL)
    if rules_match:
        rules_dict = {}
        rules_content = rules_match.group(1)

        # Find all key: "value" pairs
        # Matches keys in quotes, values in quotes (even with inner quotes)
        kv_pattern = re.compile(r'"(.*?)"\s*:\s*"((?:[^"]|"(?!\s*:))*?)"\s*(?:,|$)', re.DOTALL)
        for m in kv_pattern.finditer(rules_content):
            key = m.group(1)
            value = m.group(2)
            rules_dict[key] = value

        result['rules_applied'] = rules_dict

    return result


def process_enhancer_answer(
    lm,
    enhancer_name: str,
    info,
    answer: str,
    snippets: list,
    scores: list,
    row: pd.Series,
    index_column: str,
    row_position: int,
    answer_dict_column: str = None,
    replace_answer_with_dict: bool = False,
):
    answer_column = f"{enhancer_name}_answer"
    new_row = {
        index_column: row[index_column],
        f'{enhancer_name}_pred': np.nan,
        info['target']: row[info['target']] if info['target'] in row else np.nan,
        'step_by_step_thinking': np.nan,
        'feature_importance_ranking': np.nan,
        'rules_applied': np.nan,
        answer_column: answer,
        'snippets': snippets,
        'scores': scores
    }
    if answer_dict_column is not None:
        new_row[answer_dict_column] = np.nan

    answer_dict = None
    parse_method = None

    if isinstance(answer, dict):
        answer_dict = answer
        parse_method = "dict"
    else:
        try:
            answer_dict = safe_parse_medrag_answer(str(answer))
            parse_method = "json_safe"
        except Exception:
            try:
                answer_dict = ast.literal_eval(answer)
                parse_method = "literal_eval"
            except Exception:
                try:
                    answer_dict = parse_medrag_string(str(answer))
                    parse_method = "parse_string"
                except Exception as e:
                    lm.printl(f"Failed parse patient {row_position}: {e}")
                    new_row['parse_status'] = 'failed'
                    new_row['parse_method'] = None
                    return new_row

    if isinstance(answer_dict, dict):
        if "answer_choice" in answer_dict:
            raw_choice = answer_dict.pop("answer_choice", None)
            answer_dict["classification"] = normalize_answer_choice(raw_choice)

        new_row.update({
            f'{enhancer_name}_pred': answer_dict.get('classification', np.nan),
            'step_by_step_thinking': answer_dict.get('step_by_step_thinking', np.nan),
            'feature_importance_ranking': answer_dict.get('feature_importance_ranking', np.nan),
            'rules_applied': answer_dict.get('rules_applied', np.nan),
            'parse_status': 'success',
            'parse_method': parse_method,
        })

        if answer_dict_column is not None:
            new_row[answer_dict_column] = answer_dict
        if replace_answer_with_dict:
            new_row[answer_column] = answer_dict

    return new_row


# HuggingFace answer processing
def process_huggingface_answer(lm, enhancer_name: str, llm_name: str, info, answer: str, snippets: list, scores: list, row: pd.Series, ind: str, i: int):
    return process_enhancer_answer(
        lm=lm,
        enhancer_name=enhancer_name,
        info=info,
        answer=answer,
        snippets=snippets,
        scores=scores,
        row=row,
        index_column=ind,
        row_position=i,
        answer_dict_column=f"{enhancer_name}_answer_dict",
        replace_answer_with_dict=False,
    )



# MEDRAG answer processing
# Note: This is very similar to the HuggingFace version, but we keep it separate for clarity and in case we want to customize the parsing logic for MedRAG-specific answer formats in the future.
def process_medrag_answer(lm, info, answer: str, snippets: list, scores: list, row: pd.Series, ind: str, i: int):
    return process_enhancer_answer(
        lm=lm,
        enhancer_name="medrag",
        info=info,
        answer=answer,
        snippets=snippets,
        scores=scores,
        row=row,
        index_column=ind,
        row_position=i,
        answer_dict_column=None,
        replace_answer_with_dict=True,
    )


# def safe_parse_medrag_answer(answer: str) -> dict:
#     """
#     Extract and parse the first JSON object found in a MedRAG answer.
#     """
#     match = re.search(r'\{.*\}', answer, re.DOTALL)
#     if not match:
#         raise ValueError("No JSON object found in answer")

#     json_str = match.group(0)
#     return json.loads(json_str)


def safe_parse_medrag_answer(answer: str) -> dict:
    """
    Extract and parse the first JSON object found in a MedRAG answer.
    First tries strict JSON parsing, then applies light repairs for common LLM mistakes.
    """
    match = re.search(r'\{.*\}', answer, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in answer")

    json_str = match.group(0).strip()

    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        repaired = json_str

        # Fix common missing-comma errors before expected keys (Llama2 often misses commas before "classification" or "feature_importance_ranking")
        for key in ["classification", "feature_importance_ranking", "rules_applied"]:
            repaired = re.sub(
                rf'("\s*)"{key}"\s*:',
                rf'", "{key}":',
                repaired
            )

        return json.loads(repaired)


def normalize_answer_choice(value):
    if value is None:
        return np.nan

    # Already numeric
    if isinstance(value, (int, np.integer)):
        return int(value)

    if isinstance(value, str):
        v = value.strip().lower()

            # 🔹 Meditron / MCQ-style outputs
        if v == "a":
            return 1
        if v == "b":
            return 0

        if v in {"1", "yes", "true", "positive", "a. yes"}:
            return 1

        if v in {"0", "no", "false", "negative", "b. no"}:
            return 0

        # Explicit uncertainty
        if v in {"cannot_tell", "unknown", "uncertain", "n/a", "na"}:
            return np.nan

    # Everything else → unknown
    return np.nan

# def parse_medrag_string(answer):
#     result = {}

#     # 1. step_by_step_thinking
#     match = re.search(r'"step_by_step_thinking"\s*:\s*"(.+?)"\s*"classification"', answer, re.DOTALL)
#     if match:
#         result['step_by_step_thinking'] = match.group(1)

#     # 2. classification
#     match = re.search(r'"classification"\s*:\s*(\d+)', answer)
#     if match:
#         result['classification'] = int(match.group(1))

#     # 3. feature_importance_ranking
#     match = re.search(r'"feature_importance_ranking"\s*:\s*(\[[^\]]*\])', answer)
#     if match:
#         # convert single quotes to double quotes for JSON parsing
#         try:
#             result['feature_importance_ranking'] = [x.strip().strip('"').strip("'") 
#                                                 for x in re.findall(r"'(.*?)'|\"(.*?)\"", match.group(1))]
#         except Exception:
#             result['feature_importance_ranking'] = None

#     # 4. rules_applied using regex (no json.loads)
#     rules_match = re.search(r'"rules_applied"\s*:\s*\{(.+?)\}\s*$', answer, re.DOTALL)
#     if rules_match:
#         rules_dict = {}
#         rules_content = rules_match.group(1)

#         # Find all key: "value" pairs
#         # Matches keys in quotes, values in quotes (even with inner quotes)
#         kv_pattern = re.compile(r'"(.*?)"\s*:\s*"((?:[^"]|"(?!\s*:))*?)"\s*(?:,|$)', re.DOTALL)
#         for m in kv_pattern.finditer(rules_content):
#             key = m.group(1)
#             value = m.group(2)
#             rules_dict[key] = value

#         result['rules_applied'] = rules_dict

#     return result



# def _extract_between_keys(text: str, start_key: str, end_key: str):
#     """
#     Extract the raw value between two JSON-like keys in a malformed LLM answer.
#     Assumes the value starts after: "start_key":
#     and ends right before: "end_key"
#     """
#     start_marker = f'"{start_key}"'
#     end_marker = f'"{end_key}"'

#     start = text.find(start_marker)
#     if start == -1:
#         return None

#     # move after `"start_key":`
#     start = text.find(":", start)
#     if start == -1:
#         return None
#     start += 1

#     end = text.find(end_marker, start)
#     if end == -1:
#         return None

#     value = text[start:end].strip()

#     # remove trailing comma before next key if present
#     if value.endswith(","):
#         value = value[:-1].rstrip()

#     # remove surrounding quotes if present
#     if len(value) >= 2 and value[0] == '"' and value[-1] == '"':
#         value = value[1:-1]

#     return value.strip()

import re
import ast

def _extract_between_keys_any(text: str, start_keys: list[str], end_keys: list[str]):
    start_pos = -1
    start_key_found = None

    for start_key in start_keys:
        pos = text.find(start_key)
        if pos != -1 and (start_pos == -1 or pos < start_pos):
            start_pos = pos
            start_key_found = start_key

    if start_pos == -1:
        return None

    start = text.find(":", start_pos)
    if start == -1:
        return None
    start += 1

    end_positions = []
    for end_key in end_keys:
        pos = text.find(end_key, start)
        if pos != -1:
            end_positions.append(pos)

    if not end_positions:
        return None

    end = min(end_positions)
    value = text[start:end].strip()

    if value.endswith(","):
        value = value[:-1].rstrip()

    if len(value) >= 2 and value[0] == '"' and value[-1] == '"':
        value = value[1:-1]

    return value.strip()

# def parse_medrag_string(answer):
#     result = {}

#     # 1. step_by_step_thinking
#     step_text = _extract_between_keys(answer, "step_by_step_thinking", "classification")
#     if step_text is not None:
#         result["step_by_step_thinking"] = step_text

#     # 2. classification
#     match = re.search(r'"classification"\s*:\s*(\d+)', answer)
#     if match:
#         result["classification"] = int(match.group(1))

#     # 3. feature_importance_ranking
#     match = re.search(r'"feature_importance_ranking"\s*:\s*(\[[^\]]*\])', answer, re.DOTALL)
#     if match:
#         try:
#             result["feature_importance_ranking"] = ast.literal_eval(match.group(1))
#         except Exception:
#             result["feature_importance_ranking"] = None

#     # 4. rules_applied
#     rules_match = re.search(r'"rules_applied"\s*:\s*\{(.+?)\}\s*$', answer, re.DOTALL)
#     if rules_match:
#         rules_dict = {}
#         rules_content = rules_match.group(1)

#         kv_pattern = re.compile(
#             r'"(.*?)"\s*:\s*"((?:[^"]|"(?!\s*:))*?)"\s*(?:,|$)',
#             re.DOTALL
#         )
#         for m in kv_pattern.finditer(rules_content):
#             key = m.group(1)
#             value = m.group(2)
#             rules_dict[key] = value

#         result["rules_applied"] = rules_dict

#     return result

def parse_medrag_string(answer):
    result = {}

    # 1. step_by_step_thinking
    step_text = _extract_between_keys_any(
        answer,
        ['"step_by_step_thinking"', '"step-by-step-thinking"'],
        ['"classification"', 'classification"']
    )
    if step_text is not None:
        result["step_by_step_thinking"] = step_text

    # 2. classification
    match = re.search(r'"?classification"\s*:\s*(\d+)', answer)
    if match:
        result["classification"] = int(match.group(1))

    # 3. feature_importance_ranking

    # Case A: proper JSON-like list
    match = re.search(r'"feature_importance_ranking"\s*:\s*(\[[^\]]*\])', answer, re.DOTALL)
    if match:
        try:
            result['feature_importance_ranking'] = ast.literal_eval(match.group(1))
        except Exception:
            result['feature_importance_ranking'] = None

    # Case B: malformed plain-text list embedded in the answer text, e.g.
    # feature_importance_ranking: [age, hypertension, heart disease, ...]
    if "feature_importance_ranking" not in result or result["feature_importance_ranking"] is None:
        match = re.search(
            r'feature_importance_ranking\s*:\s*\[([^\]]+)\]',
            answer,
            re.DOTALL
        )
        if match:
            raw_items = match.group(1)
            items = [x.strip() for x in raw_items.split(",")]
            result["feature_importance_ranking"] = [x for x in items if x]

    # 4. rules_applied ...

    return result
