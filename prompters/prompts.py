"""
Emotion Classification Prompts for the EmotionRec system.
"""

##################### EMOTION CLASSIFICATION PROMPTS #####################

# ED_hard_a dataset: 4 emotion labels with index mapping
LABEL2IDX = {
    'anxious': 0,
    'apprehensive': 1,
    'afraid': 2,
    'terrified': 3,
}
IDX2LABEL = {v: k for k, v in LABEL2IDX.items()}
EMOTION_LIST = list(LABEL2IDX.keys())

emotion_user_prompt = """\
Analyze and think through the emotional content of the following text step by step.
Reflect on the cues and context provided to determine the most fitting emotion from the given list.

Your task:
1) Provide 2-3 concise sentences explaining the cues and context.
2) Output the final emotion wrapped in {emb_token} and {emb_end_token}.

Choose one emotion from this list:
anxious, apprehensive, afraid, terrified

IMPORTANT:
- Keep the reasoning brief (2-3 sentences total).
- The final {emb_token}emotion{emb_end_token} tag must be the LAST thing in your response.

Text:
"""

emotion_item_prompt = """\
Describe the characteristics of the emotion "{{emotion_name}}" inside {{emb_token}} and {{emb_end_token}}.
What feelings, situations, or expressions are typically associated with this emotion?\
"""


def obtain_prompts():
    """Obtain prompts for emotion classification."""
    return {
        "user_prompt": emotion_user_prompt,
        "item_prompt": emotion_item_prompt,
    }
