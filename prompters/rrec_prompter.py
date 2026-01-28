"""
Emotion Classification Prompters for the EmotionRec system.
Converts emotion datasets to prompts for LLM processing.
"""
from typing import Optional
import datasets
from prompters.abstract_prompter import AbstractPrompter
from prompters.prompts import obtain_prompts


def get_emotion_text(sequence):
    """Get the text to classify for emotion detection."""
    if 'text' in sequence and sequence['text']:
        return sequence['text']
    if 'history_item_title' in sequence and sequence['history_item_title']:
        return sequence['history_item_title'][0]
    return ""


def get_emotion_info_str(sequence, just_title=False):
    """Get emotion label info for ItemPrompter."""
    emotion_name = sequence.get('title', sequence.get('item_title', ''))
    if just_title:
        return emotion_name
    return f"Emotion: {emotion_name}"


class UserGenPrompter(AbstractPrompter):
    """Prompter for generating user emotion analysis prompts."""
    user_content_key = "prompt"
    sys_content_key = None

    def __init__(self,
                 tokenizer,
                 dset=None,
                 input_ids_max_length=2048,
                 emb_token='',
                 emb_end_token='',
                 ):
        super().__init__(tokenizer)
        self.dset = dset
        self.input_ids_max_length = input_ids_max_length
        self.emb_token = emb_token or getattr(self.tokenizer, 'generation_end', self.tokenizer.eos_token)
        self.emb_end_token = emb_end_token

        prompts = obtain_prompts()
        self.user_analyze_prompt = prompts['user_prompt'].format(
            emb_token=self.emb_token,
            emb_end_token=self.emb_end_token)

    def to_chat_example(self, sequence):
        """Convert sequence to chat example with emotion text."""
        text = get_emotion_text(sequence)
        sequence['prompt'] = self.user_analyze_prompt + '\n' + text
        return sequence

    def convert_dataset(self,
                        split: Optional[str] = None,
                        dset: datasets.Dataset = None,
                        return_messages: bool = False,
                        ):
        new_dataset = self.dset[split] if split is not None else dset

        new_dataset = new_dataset.map(self.to_chat_example,
                                      desc='Converting to chat examples',
                                      batched=False,
                                      keep_in_memory=True)

        if return_messages:
            def _to_messages(example):
                example['messages'] = self.formatting_func(example, return_messages=True)
                return example
            return new_dataset.map(_to_messages,
                                   desc='Converting to chat examples',
                                   batched=False)

        new_dataset = new_dataset.map(self.totensor,
                                      desc='Applying chat template',
                                      batched=True,
                                      fn_kwargs={'max_length': self.input_ids_max_length})
        
        if 'user_gen_input_ids' in new_dataset.column_names:
            new_dataset = new_dataset.remove_columns(['user_gen_input_ids', 'user_gen_attention_mask'])

        new_dataset = new_dataset.rename_column('input_ids', 'user_gen_input_ids')
        new_dataset = new_dataset.rename_column('attention_mask', 'user_gen_attention_mask')

        return new_dataset


class ItemPrompter(AbstractPrompter):
    """Prompter for generating emotion embedding prompts."""
    user_content_key = "prompt"
    assistant_content_key = "assistant"
    
    def __init__(self,
                 tokenizer,
                 dset=None,
                 input_ids_max_length=512,
                 emb_token='',
                 emb_end_token='',
                 ):
        super().__init__(tokenizer)
        self.dset = dset
        self.input_ids_max_length = input_ids_max_length
        self.emb_token = emb_token or getattr(self.tokenizer, 'generation_end', self.tokenizer.eos_token)
        self.emb_end_token = emb_end_token
        self.prompts = obtain_prompts()

    def convert_dataset(self,
                        split: Optional[str] = None,
                        dset: datasets.Dataset = None,
                        ):
        new_dataset = self.dset[split] if split is not None else dset

        new_dataset = new_dataset.map(self.to_chat_example_item,
                                      desc='Converting to chat examples for emotion profile',
                                      batched=False,
                                      keep_in_memory=True)
        
        new_dataset = new_dataset.map(self.totensor,
                                      desc='Applying chat template for emotion profile',
                                      batched=True,
                                      fn_kwargs={
                                          'max_length': self.input_ids_max_length,
                                          'continue_final_message': True,
                                      })
        
        if 'item_input_ids' in new_dataset.column_names:
            new_dataset = new_dataset.remove_columns(['item_input_ids', 'item_attention_mask'])
        
        new_dataset = new_dataset.rename_column('input_ids', 'item_input_ids')
        new_dataset = new_dataset.rename_column('attention_mask', 'item_attention_mask')

        return new_dataset

    def to_chat_example_item(self, sequence):
        """Convert emotion to chat example - directly embed the label."""
        emotion_name = sequence.get('title', sequence.get('item_title', ''))
        # Enrich label with a short description to reduce embedding collapse.
        desc_map = {
            "anxious": "worry about a future or uncertain event",
            "apprehensive": "uneasy expectation of something bad",
            "afraid": "fear of immediate danger or threat",
            "terrified": "intense fear or panic",
            "sad": "feeling down or sorrowful",
            "joyful": "feeling happy and pleased",
            "angry": "feeling mad or irritated",
        }
        desc = desc_map.get(str(emotion_name).strip().lower(), "the primary emotion described")
        sequence['prompt'] = (
            f"Emotion label: {emotion_name}. "
            f"Description: {desc}. "
            f"Output: {self.emb_token}{emotion_name}"
        )
        sequence['assistant'] = self.emb_token
        return sequence


class UserPrompter(AbstractPrompter):
    """Prompter for processing user profiles with generated emotion reasoning."""
    assistant_content_key = 'completion_0'
    user_content_key = "prompt"
    sys_content_key = None

    def __init__(self,
                 tokenizer,
                 dset=None,
                 input_ids_max_length=1024,
                 emb_token='',
                 emb_end_token='',
                 ):
        super().__init__(tokenizer)
        self.dset = dset
        self.input_ids_max_length = input_ids_max_length
        self.emb_token = emb_token or getattr(self.tokenizer, 'generation_end', self.tokenizer.eos_token)
        self.emb_end_token = emb_end_token

        self.prompts = obtain_prompts()
        self.prompts['user_prompt'] = self.prompts['user_prompt'].format(
            emb_token=self.emb_token,
            emb_end_token=self.emb_end_token,
        )

    def convert_dataset(self,
                        split: Optional[str] = None,
                        dset: datasets.Dataset = None,
                        ):
        new_dataset = self.dset[split] if split is not None else dset

        assert split is None or split != 'item_info'

        new_dataset = new_dataset.map(self.to_chat_example,
                                      desc='Converting to chat examples for user profile',
                                      batched=False,
                                      keep_in_memory=True)
        
        new_dataset = new_dataset.map(self.totensor,
                                      desc='Applying chat template for user profile',
                                      batched=True,
                                      fn_kwargs={
                                          'max_length': self.input_ids_max_length,
                                          'continue_final_message': True
                                      })
        
        if 'user_input_ids' in new_dataset.column_names:
            new_dataset = new_dataset.remove_columns(['user_input_ids', 'user_attention_mask'])
        
        new_dataset = new_dataset.rename_column('input_ids', 'user_input_ids')
        new_dataset = new_dataset.rename_column('attention_mask', 'user_attention_mask')

        return new_dataset

    def to_chat_example(self, sequence):
        """Convert sequence to chat example with emotion text and profile."""
        text = get_emotion_text(sequence)
        prompt = self.prompts["user_prompt"]
        sequence["prompt"] = prompt + "\n" + text

        if isinstance(sequence['profile'], str):
            sequence['profile'] = [sequence['profile']]
        
        if isinstance(sequence['profile'], list):
            for i, profile in enumerate(sequence['profile']):
                generation_end = getattr(self.tokenizer, 'generation_end', None)
                if self.emb_token != generation_end:
                    if not profile.rstrip().endswith(self.emb_token):
                        profile = profile + self.emb_token
                sequence[f"completion_{i}"] = profile
        else:
            raise ValueError(f"profile is not a string or list: {sequence['profile']}")
        return sequence

    def totensor_multiple(self, element):
        """Convert multiple completions to tensors."""
        outputs = self.tokenizer(
            [self.formatting_func(element,
                                  completions_key=f"completion_{i}",
                                  continue_final_message=True
                                  ) for i in range(len(element['profile']))],
            add_special_tokens=False,
            truncation=True,
            padding=False,
            max_length=self.input_ids_max_length,
            return_overflowing_tokens=False,
            return_length=False,
        )

        result = {
            "multi_user_input_ids": outputs["input_ids"],
            "multi_user_attention_mask": outputs["attention_mask"],
        }
        result["multi_user_completion_range"] = self.find_completion_start_end(outputs["input_ids"])
        return result
