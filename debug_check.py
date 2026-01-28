#!/usr/bin/env python
from datasets import load_from_disk

# Load the test dataset with generated completions
ds = load_from_disk('../checkpoints/emotion-rl-v15/datasets/reasoning_test_0')
print(f'Dataset has {len(ds)} examples')
print(f'Columns: {ds.column_names}')
print()

# Check first 10 examples' profile (generated completion)
for i in range(min(10, len(ds))):
    example = ds[i]
    profile = example.get('profile', 'NO PROFILE')
    emotion = example.get('emotion_label', 'NO LABEL')
    print(f'Example {i}: label={emotion}')
    print(f'  profile="{profile}"')
    print()
