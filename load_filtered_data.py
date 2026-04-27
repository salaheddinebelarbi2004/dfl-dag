"""
Simple data loader: Load Excel, remove first 4000 spam + 4000 ham, save CSV
No embeddings - just raw data filtering
"""

import pandas as pd
import numpy as np
import sys

# Force UTF-8 output
sys.stdout.reconfigure(encoding='utf-8')

print("[LOAD] Loading Excel file...")
df = pd.read_excel("Augmented_SpamHamTweets.xlsx")

print(f"[INFO] Total rows: {len(df)}")
print(f"[INFO] Columns: {df.columns.tolist()}")

# Get label column (should be 'Label')
label_col = 'Label'
message_col = 'Tweet Text'

# Check unique values
print(f"[INFO] Label value counts:")
print(df[label_col].value_counts())

spam = df[df[label_col] == 'Spam']
ham = df[df[label_col] == 'Ham']

print(f"\n[STATS] BEFORE filtering:")
print(f"   Spam: {len(spam)}")
print(f"   Ham: {len(ham)}")
print(f"   Total: {len(df)}")

# Remove first 4000 of each
spam_new = spam.iloc[4000:]
ham_new = ham.iloc[4000:]

df_new = pd.concat([spam_new, ham_new], ignore_index=True)
df_new = df_new.sample(frac=1).reset_index(drop=True)

print(f"\n[STATS] AFTER removing first 4000 spam + 4000 ham:")
print(f"   Spam: {len(spam_new)}")
print(f"   Ham: {len(ham_new)}")
print(f"   Total: {len(df_new)}")

# Save
df_new.to_csv("filtered_training_data.csv", index=False, encoding='utf-8')
print(f"\n[OK] Data saved to: filtered_training_data.csv")
print(f"[OK] Ready for embedding generation!")
