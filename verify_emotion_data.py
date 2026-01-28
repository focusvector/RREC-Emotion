from datasets import load_from_disk

ds = load_from_disk("data/ED_easy_4_processed")

print("Dataset structure:")
print(ds)
print("\nTrain sample:")
print(ds["train"][0])
print("\nItem info sample:")
print(ds["item_info"][:5])
