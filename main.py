import praw
import pandas as pd

reddit = praw.Reddit(
    client_id="zcdbVMZf_NegxrN30xvPxQ",
    client_secret="mTWF57bu-7NRORuCtrlKxR2-I74axA",
    password="Springer!!",
    user_agent="testscript by u/unrodder",
    username="unrodder",
)

subreddit = reddit.subreddit("SpainEconomics")

# Create a list to store post data
posts_data = []

# Fetch the 100 most recent posts
for post in subreddit.new(limit=None):
    # Check if the post is text-only (not an image, video, or link post)
    if not post.is_self:
        continue

    # Extract relevant information
    posts_data.append({
        "title": post.title,
        "text": post.selftext,
        "score": post.score,
        "num_comments": post.num_comments,
        "created_utc": post.created_utc
    })

# Convert to DataFrame
df = pd.DataFrame(posts_data)
# Display the first few rows
print(df.head())

# Save to CSV (optional)
df.to_csv("spain_subreddit_posts.csv", index=False)

print(f"Collected {len(df)} text-only posts from r/Spain")

