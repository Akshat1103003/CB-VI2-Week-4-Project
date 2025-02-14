# YouTube Data Analysis - Netflix India

## Overview

This project analyzes YouTube video data from the Netflix India channel to derive insights into factors affecting video performance, audience engagement, and content strategy.

## Dataset

- **Source:** YouTube API data
- **File:** `netflix_india_yt_data.csv`
- **Key Columns:** `video_id`, `title`, `duration`, `viewCount`, `likeCount`, `commentCount`, `publishedAt`, `tags`

---

## Questions & Insights

### 1. Does the duration of a video influence views and comments?

- **Method:** Converted ISO 8601 duration into total minutes.

```python
import pandas as pd
import re

df = pd.read_csv("netflix_india_yt_data.csv") # Load your CSV file
dfdata = {'duration': ['PT26M12S', 'PT1H26M10S', 'PT45S', 'PT2H', 'PT1H', 'PT1M']}
df2 = df["duration"]

def duration_to_minutes(duration):
    hours = re.search(r'(\d+)H', duration)
    minutes = re.search(r'(\d+)M', duration)
    seconds = re.search(r'(\d+)S', duration)

    h = int(hours.group(1)) if hours else 0
    m = int(minutes.group(1)) if minutes else 0
    s = int(seconds.group(1)) if seconds else 0

    total_minutes = h * 60 + m + s / 60
    return round(total_minutes, 2)

df['total_minutes'] = df['duration'].apply(duration_to_minutes)

print(df[['duration', 'total_minutes']].head()) # Display the new column
```
Findings:
- Correlation between viewCount & duration = 0.001 (no impact).
- Correlation between commentCount & duration = 0.056 (no impact).

### 2. Is there a relationship between views and comments?
- Correlation found: 0.143 (weak relationship).
- Conclusion: Views do not necessarily translate into more comments.
### 3. Does the number of tags impact video views?
- Correlation between tags count & views: 0.008 (no impact).
- Conclusion: The number of tags does not significantly affect views.
### 4. Does publishing time influence engagement?
- Method: Extracted day of the week & time of day from publishedAt.
```
df["day_of_week"] = df["publishedAt"].dt.day_name()
df["publishedAt"] = pd.to_datetime(df["publishedAt"])

df["published_hour"] = df["publishedAt"].dt.hour

def label_time(hour):
    if 5 <= hour < 12:
        return "Morning"
    elif 12 <= hour < 17:
        return "Afternoon"
    elif 17 <= hour < 21:
        return "Evening"
    else:
        return "Night"

df["time_of_day"] = df["published_hour"].apply(label_time)

print(df[["publishedAt", "published_hour", "time_of_day"]].head())
print(df.groupby("time_of_day")["viewCount"].count())
```
#### Findings:
- Most videos are published in Morning & Afternoon.
- Time of publishing does impact engagement levels.
### 5. What is the most popular video on our channel & why?
- Identified the video with the highest views.
- Factors analyzed: Likes, comments, publish time, duration.

```
most_popular_video = df.loc[df["viewCount"].idxmax()]
print(most_popular_video[["video_id", "title", "viewCount", "likeCount", "commentCount", "publishedAt", "duration"]])
```
### 6. Does title length affect video views ?
- Correlation between title length & views: 0.003 (no impact).
- Conclusion: Title length does not influence video popularity.
```
df["title_length"] = df["title"].str.len()
print(df.corr(numeric_only=True))
```
### 7. What is the distribution of video views?
#### Findings :
- Categorized videos based on view count:
- Viral (>1M): 1,037 videos
- High (100K-1M): 2,105 videos
- Medium (10K-100K): 1,745 videos
- Low (<10K): 52 videos
```
df["view_category"] = pd.cut(
  df["viewCount"], bins=[0, 10000, 100000, 1000000, 10000000],
  labels=["Low (<10K)", "Medium (10K-100K)", "High (100K-1M)", "Viral (>1M)"],category_counts = df["view_category"].value_counts()
)
plt.figure(figsize=(8, 5))
ax = category_counts.plot(kind="bar", color=["red", "blue", "green", "purple"])

for i, count in enumerate(category_counts):
    plt.text(i, count + 5, str(count), ha='center', va='bottom', fontsize=10, fontweight='bold', color='black')

plt.title("Number of Videos in Each View Category")
plt.xlabel("View Category")
plt.ylabel("Count of Videos")
plt.xticks(rotation=45)
plt.show()
```
### 8. Are certain topics/themes consistently performing better?
- Method: Extracted & analyzed most common tags.
- Findings: Identified top-performing video themes based on average views.
```
from collections import Counter
import numpy as np

df["tags"] = df["tags"].apply(lambda x: eval(x) if isinstance(x, str) else x)
all_tags = [tag for tags_list in df["tags"].dropna() for tag in tags_list]

common_tags = Counter(all_tags).most_common(20)
common_tags_df = pd.DataFrame(common_tags, columns=["Theme", "Count"])

df_exploded = df.explode("tags")

theme_performance = df_exploded.groupby("tags").agg(
    avg_views=("viewCount", np.mean),
    avg_likes=("likeCount", np.mean),
    avg_comments=("commentCount", np.mean),
    video_count=("video_id", "count")
).reset_index()

theme_performance = theme_performance.sort_values(by="avg_views", ascending=False)

plt.figure(figsize=(12, 6))
top_themes = theme_performance.head(10)

sns.barplot(x=top_themes["avg_views"], y=top_themes["tags"], palette="coolwarm")
plt.xlabel("Average Views")
plt.ylabel("Theme")
plt.title("Top 10 Performing Video Themes (by Views)")
plt.show()
```
### 9. Are there any high-performing videos with no tags?
- Findings: No videos with zero tags had more than 1M views.

### 10. What is the frequency of video publishing?
- Method: Monthly analysis of publishedAt.
```
df["published_month"] = df["publishedAt"].dt.to_period("M")

df["published_month"].value_counts().sort_index().plot(kind="bar", color="purple", figsize=(40, 6))
plt.title("Monthly Video Publishing Frequency")
plt.xlabel("Month")
plt.ylabel("Number of Videos Published")
plt.xticks(rotation=45)
plt.show()
```
### 11. What are the most used keywords in video tags?
- Extracted top keywords from the tags column.
- Top keywords: Netflix, Netflix India, Trailer, Comedy, Bollywood.
```
from collections import Counter
import itertools

df["tags_normalized"] = df["tags"].apply(lambda x: [tag.lower() for tag in x] if isinstance(x, list) else [])
all_tags = list(itertools.chain(*df["tags_normalized"]))
tag_counts = Counter(all_tags)
tag_df = pd.DataFrame(tag_counts.items(), columns=["Keyword", "Count"]).sort_values(by="Count", ascending=False)
df["tags_normalized"]
```
### Q12. What is the most used keyword in YT titles ?
- #### Findings : The most used keywords in Youtube Titles are :
- Netflix
- India
- the trailer
- official
- ft.
