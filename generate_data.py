import json
import os

# We need 30+ articles with some duplicates to test the logic
mock_data = [
    # Event 1: HDFC Dividend (Semantic Duplicates)
    {
        "id": "1",
        "text": "HDFC Bank announces 15% dividend, board approves stock buyback",
        "source": "NSE India",
        "published_at": "2023-10-25T10:00:00Z"
    },
    {
        "id": "2",
        "text": "Board of HDFC Bank approves buyback of shares and declares 15% dividend payout.",
        "source": "MoneyControl",
        "published_at": "2023-10-25T10:05:00Z"
    },
    # Event 2: RBI Repo Rate (Direct Duplicate Concept)
    {
        "id": "3",
        "text": "RBI raises repo rate by 25bps to 6.75%, citing inflation concerns",
        "source": "RBI Press Release",
        "published_at": "2023-10-26T09:00:00Z"
    },
    {
        "id": "4",
        "text": "Reserve Bank hikes interest rates by 0.25% in surprise move to fight inflation.",
        "source": "Bloomberg",
        "published_at": "2023-10-26T09:15:00Z"
    },
    # Event 3: Banking Sector (Contextual Search)
    {
        "id": "5",
        "text": "Banking sector NPAs decline to 5-year low, credit growth at 16%",
        "source": "Economic Times",
        "published_at": "2023-10-27T11:00:00Z"
    }
]

# Add filler articles to reach 30 count requirement
for i in range(6, 35):
    mock_data.append({
        "id": str(i),
        "text": f"Market update {i}: Sensex rises by {i*10} points driven by tech rally.",
        "source": "Reuters",
        "published_at": "2023-10-28T09:00:00Z"
    })

# Ensure data directory exists
os.makedirs("data", exist_ok=True)

# Write to JSON
with open("data/mock_news.json", "w") as f:
    json.dump(mock_data, f, indent=4)

print(f"Successfully generated {len(mock_data)} articles in data/mock_news.json")