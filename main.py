import json
import tiktoken  # for token counting
import numpy as np
from collections import defaultdict
import ast
import os
import createModel as cm


def getJSON():
    data_path = "mydata.jsonl"
    ds = ""
    with open(data_path, "r", encoding="utf-8") as f:
        data = f.readlines()
        for line in data:
            ds += line + "\n"
    dataset = json.loads(ds)

    return dataset


def checkErrors(dataset):
    return ast.literal_eval(json.dumps(dataset))


def dataSetsStats(dataset):
    # Initial dataset stats
    print("Num examples:", len(dataset))
    print("First example:")
    count = 1

    for message in dataset["messages"]:
        print("messages{}".format(str(count)))
        print(message)
        count += 1

    # Format error checks
    format_errors = defaultdict(int)

    count = 1

    for ex in dataset:
        if not isinstance(ex, dict):
            format_errors["data_type"] += 1
            continue

        messages = ex.get("messages", None)
        count += 1
        if not messages:
            format_errors["missing_messages_list"] += 1
            continue

        for message in messages:
            if "role" not in message or "content" not in message:
                format_errors["message_missing_key"] += 1

            if any(
                k not in ("role", "content", "name", "function_call", "weight")
                for k in message
            ):
                format_errors["message_unrecognized_key"] += 1

            if message.get("role", None) not in (
                "system",
                "user",
                "assistant",
                "function",
            ):
                format_errors["unrecognized_role"] += 1

            content = message.get("content", None)
            function_call = message.get("function_call", None)

            if (not content and not function_call) or not isinstance(content, str):
                format_errors["missing_content"] += 1

        if not any(message.get("role", None) == "assistant" for message in messages):
            format_errors["example_missing_assistant_message"] += 1

    if format_errors:
        print("Found errors:")
        for k, v in format_errors.items():
            print(f"{k}: {v}")
    else:
        print("No errors found")


def num_tokens_from_messages(messages, tokens_per_message=3, tokens_per_name=1):
    encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3
    return num_tokens


def num_assistant_tokens_from_messages(messages, encoding):
    num_tokens = 0
    for message in messages:
        if message["role"] == "assistant":
            num_tokens += len(encoding.encode(message["content"]))
    return num_tokens


def print_distribution(values, name):
    print(f"\n#### Distribution of {name}:")
    print(f"min / max: {min(values)}, {max(values)}")
    print(f"mean / median: {np.mean(values)}, {np.median(values)}")
    print(f"p5 / p95: {np.quantile(values, 0.1)}, {np.quantile(values, 0.9)}")


def warningsAndTokensCounts(dataset, state):
    # Warnings and tokens counts
    encoding = tiktoken.get_encoding("cl100k_base")

    n_missing_system = 0
    n_missing_user = 0
    n_messages = []
    convo_lens = []
    assistant_message_lens = []
    count = 1
    for ex in dataset:
        messages = ex["messages"]
        count += 1
        if not any(message["role"] == "system" for message in messages):
            n_missing_system += 1
        if not any(message["role"] == "user" for message in messages):
            n_missing_user += 1
        n_messages.append(len(messages))
        convo_lens.append(num_tokens_from_messages(messages))
        assistant_message_lens.append(
            num_assistant_tokens_from_messages(messages, encoding)
        )
    if state:
        print("Num examples missing system message:", n_missing_system)
        print("Num examples missing user message:", n_missing_user)
        print_distribution(n_messages, "num_messages_per_example")
        print_distribution(convo_lens, "num_total_tokens_per_example")
        print_distribution(assistant_message_lens, "num_assistant_tokens_per_example")
        n_too_long = sum(l > 4096 for l in convo_lens)  # noqa: E741
        print(
            f"\n{n_too_long} examples may be over the 4096 token limit, they will be truncated during fine-tuning"
        )
    return convo_lens


def costAffectiveness(dataset, convo_lens):

    # Pricing and default n_epochs estimate
    MAX_TOKENS_PER_EXAMPLE = 4096

    TARGET_EPOCHS = 3
    MIN_TARGET_EXAMPLES = 100
    MAX_TARGET_EXAMPLES = 25000
    MIN_DEFAULT_EPOCHS = 1
    MAX_DEFAULT_EPOCHS = 25

    n_epochs = TARGET_EPOCHS
    n_train_examples = len(dataset)
    if n_train_examples * TARGET_EPOCHS < MIN_TARGET_EXAMPLES:
        n_epochs = min(MAX_DEFAULT_EPOCHS, MIN_TARGET_EXAMPLES // n_train_examples)
    elif n_train_examples * TARGET_EPOCHS > MAX_TARGET_EXAMPLES:
        n_epochs = max(MIN_DEFAULT_EPOCHS, MAX_TARGET_EXAMPLES // n_train_examples)

    n_billing_tokens_in_dataset = sum(
        min(MAX_TOKENS_PER_EXAMPLE, length) for length in convo_lens
    )
    print(
        f"Dataset has ~{n_billing_tokens_in_dataset} tokens that will be charged for during training"
    )
    print(f"By default, you'll train for {n_epochs} epochs on this dataset")
    print(
        f"By default, you'll be charged for ~{n_epochs * n_billing_tokens_in_dataset} tokens"
    )


os.system("cls")
print("==================================")
print("=========PRE MODEL BUILDS=========")
print("==================================")
print("1) Check Errors")
print("2) Data Stat Errors")
print("3) Warnings and tokens counts")
print("4) Cost Affectiveness")
print("----------")
print("5) Create Model")
print("----------")
print("Q) Quit")
user = input("Make a selection: ")
match (user):
    case "1":
        data = getJSON()
        checkErrors(data)
    case "2":
        data = getJSON()
        dataSetsStats(data)
    case "3":
        data = getJSON()
        convo = warningsAndTokensCounts(data, True)
    case "4":
        data = getJSON()
        convo = warningsAndTokensCounts(data, False)
        costAffectiveness(data, convo)
    case "5":
        cm.createTrainingJob()
    case _:
        print("Exiting...")
