import re


def preprocess_request_dict(request_dict):
    query_passage = []
    query = request_dict["query"]
    for cand in request_dict["candidates"]:
        query_passage.append((query, cand["doc"]["segment"]))
    return query_passage


def generate_prompts(query_passage, prompt_examples, prompt_template):
    prompts = []
    for q_p in query_passage:
        prompt = prompt_template.format(
            examples=prompt_examples,
            query=q_p[0],
            passage=q_p[1],
        )
        prompts.append(prompt)
    return prompts


def parse_fewshot_response(response: str, passage: str, query: str) -> int:
    response = response.strip().lower()
    valid_res = 1
    answer = ""
    patterns = [
        r'"o": (0|1|2|3)',
        r'"overall_score": (0|1|2|3)',
        r'"overall": (0|1|2|3)',
        r'"overall score": (0|1|2|3)',
        r'"final score": (0|1|2|3)',
        r'"final_score": (0|1|2|3)',
        r'"score": (0|1|2|3)',
        r'"o_score": (0|1|2|3)',
    ]
    for pattern in patterns:
        matched = re.search(pattern, response, re.IGNORECASE | re.MULTILINE | re.DOTALL)

        if matched:
            answer = matched.group(1).capitalize()
            break
    if answer == "":
        answer = "0"
        valid_res = 0
        print(f"Invalid response to `{query}` & `{passage}`: {response}")
    return int(answer), valid_res


def prepare_judgments(outputs, query_passage, prompts, model_name):
    judgments = []
    for output, (query, passage), prompt in zip(outputs, query_passage, prompts):
        judgment = {
            "model": model_name,
            "query": query,
            "passage": passage,
            "prompt": prompt,
            "predition": output,
            "judgment": parse_fewshot_response(output, query, passage)[0],
        }
        judgments.append(judgment)
    return judgments
