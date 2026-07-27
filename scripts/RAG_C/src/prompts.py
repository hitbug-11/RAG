MULTIPLE_PROMPT = ' You are a helpful assistant, below is a query from a user and some relevant contexts. Answer the question given the information in those contexts. \nContexts: [context] \nQuestion: [question] and give the reasons?'

MULTIPLE_PROMPT_CoT = ' Question: [question] and Why? Could you provide two distinct reasons using the Chain-of-Thought approach? Additionally, ensure that each reason is independent and fully self-contained.'

MULTIPLE_PROMPT_Watermark = "I have a sentence: [CoT] Please give me a very rarely used phrase consisting of 2-10 rare words(constraints). The phrase should not change the meaning if it is appended to the end of these sentence. (objective). Please just output the phrase"
MULTIPLE_PROMPT_Watermark_front = "I have a sentence: [CoT] Please give me a very rarely used phrase consisting of 2-10 rare words(constraints). The phrase should not change the meaning if it is appended to the front of these sentence. (objective). Please just output the phrase"

MULTIPLE_PROMPT_CoT_M = "I have a sentence: [CoT] Please help process the sentence using third person pronoun to replace all subjects. Please just output the processed sentence"
# watermark = "ephemeral, cryptic echoes obfuscate perception."
def wrap_prompt(question, context, prompt_id=1) -> str:
    if prompt_id == 4:
        assert type(context) == list
        context_str = "\n".join(context)
        input_prompt = MULTIPLE_PROMPT.replace('[question]', question).replace('[context]', context_str)
    else:
        input_prompt = MULTIPLE_PROMPT.replace('[question]', question).replace('[context]', context)
    return input_prompt


def wrap_prompt_watermark(CoT, prompt_id=1) -> str:

    phrase_watermark = MULTIPLE_PROMPT_Watermark.replace('[CoT]', CoT)
    return phrase_watermark

def wrap_prompt_watermark_front(CoT, prompt_id=1) -> str:

    phrase_watermark = MULTIPLE_PROMPT_Watermark_front.replace('[CoT]', CoT)
    return phrase_watermark

def wrap_prompt_phrase(CoT, prompt_id=1) -> str:

    phrase_watermark = MULTIPLE_PROMPT_CoT_M.replace('[CoT]', CoT)
    return phrase_watermark


def wrap_prompt_CoT(question, context, prompt_id=1) -> str:
    if prompt_id == 4:
        assert type(context) == list
        context_str = "\n".join(context)
        input_prompt = MULTIPLE_PROMPT_CoT.replace('[question]', question)
    else:
        input_prompt = MULTIPLE_PROMPT_CoT.replace('[question]', question)
    return input_prompt


# def wrap_prompt(question, context, prompt_id=1) -> str:
#     if prompt_id == 4:
#         assert type(context) == list
#         context_str = "\n".join(context)
#         input_prompt = MULTIPLE_PROMPT.replace('[question]', "GGGG"+ question).replace('[context]', context_str)
#     else:
#         input_prompt = MULTIPLE_PROMPT.replace('[question]', "GGGG" + question).replace('[context]', context)
#     return input_prompt

