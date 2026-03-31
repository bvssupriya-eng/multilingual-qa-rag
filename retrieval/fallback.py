import requests
import urllib.parse


HEADERS = {
    "User-Agent": "MultilingualQAAssistant/1.0 (research project)"
}


def wikipedia_fallback(query, language="en"):
    try:
        # Step 1: Search Wikipedia
        search_url = f"https://{language}.wikipedia.org/w/api.php"

        search_params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "format": "json"
        }

        search_response = requests.get(
            search_url,
            params=search_params,
            headers=HEADERS,
            timeout=5
        )

        if search_response.status_code != 200:
            print("Search API failed:", search_response.status_code)
            return None

        search_data = search_response.json()

        if not search_data.get("query") or not search_data["query"]["search"]:
            print("No search results found.")
            return None

        best_title = search_data["query"]["search"][0]["title"]

        # Step 2: Fetch summary
        encoded_title = urllib.parse.quote(best_title)

        summary_url = f"https://{language}.wikipedia.org/api/rest_v1/page/summary/{encoded_title}"

        summary_response = requests.get(
            summary_url,
            headers=HEADERS,
            timeout=5
        )

        if summary_response.status_code == 200:
            summary_data = summary_response.json()

            return {
                "source": "external",
                "title": summary_data.get("title", ""),
                "text": summary_data.get("extract", "")
            }

        print("Summary API failed:", summary_response.status_code)
        return None

    except Exception as e:
        print("Wikipedia fallback error:", e)
        return None
