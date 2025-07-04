from ollama import chat
from ollama import ChatResponse
import requests
from bs4 import BeautifulSoup
import re
import copy


def fetch_article_text(url):
    # based on:
    # https://www.scrapingbee.com/blog/how-to-scrape-all-text-from-a-website-for-llm-ai-training/
    # https://www.geeksforgeeks.org/python/implementing-web-scraping-python-beautiful-soup/
    # https://stackoverflow.com/questions/68014275/scraping-an-url-using-beautifulsoup
    # TODO: Improve search filter.
    # This fetch_article_text must be improved some teerms are appearning:
    # 6 min read, Privacy Policy\nPrivacy & Cookie Settings\nMore Info\nRecommended Stories
    try:  # standard fetcher

        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/115.0.0.0 Safari/537.36"
                   }
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        article_tag = soup.find('article')
        if article_tag:
            text = article_tag.get_text(separator='\n', strip=True)
            if len(text) > 200:  # heuristic for yahoo finance
                return text
        common_classes = ['content',
                          'article-body',
                          'post-content',
                          'main-content',
                          'entry-content'
                          ]
        for class_name in common_classes:
            container = soup.find('div', class_=class_name)
            if container:
                text = container.get_text(separator='\n', strip=True)
                if len(text) > 200:  # heuristic for yahoo finance
                    return text
        # last try
        divs = soup.find_all('div')
        max_p = 0
        main_div = None
        for div in divs:
            p_count = len(div.find_all('p'))
            if p_count > max_p:
                max_p = p_count
                main_div = div
        if main_div:
            text = main_div.get_text(separator='\n', strip=True)
            if len(text) > 200:  # heuristic for yahoo finance
                return text
        return 'No information was found.'

    except Exception as e:
        return f"Error fetching article: {e}"


def extract_clean_urls(text, url_pattern):
    raw_urls = re.findall(url_pattern, text)
    # depending on database
    clean_urls = [url.rstrip('.,') for url in raw_urls]
    return clean_urls


def remove_links(text, pattern=r'https?://[^\s]+'):
    return re.sub(pattern, '', text).strip()


def prepare_prompt(text, pattern=r'https?://[^\s]+'):
    urls = extract_clean_urls(text, pattern)
    if len(urls) == 0:
        return text
    results_text = ""
    for idx, url in enumerate(urls):
        results_text += f"Web Url {str(idx+1)}: {fetch_article_text(url)}\n\n"
    # merge URL info with original texts
    cleaned_text = re.sub(pattern, results_text, text).strip()
    return cleaned_text


def extract_metrics(response, normalized_term=1e9):
    return {
        "total_duration_sec": response.total_duration / normalized_term,
        "load_duration_sec": response.load_duration / normalized_term,
        "prompt_eval_count": response.prompt_eval_count,
        "prompt_eval_duration_sec": response.prompt_eval_duration / normalized_term,
        "eval_count": response.eval_count,
        "eval_duration_sec": response.eval_duration / normalized_term,
        "response_length_chars": len(response.message.content),
    }


def runLLM(messages: list[str],
           think: bool = False,
           model: str = 'llama3.2',
           url_fetch_support: bool = True,
           options: dict = {"temperature": 0.7,
                            "top_k": 50,
                            "top_p": 0.9,
                            "tfs_z": 1.5,
                            "repeat_penalty": 1.2,
                            "mirostat": 1.0
                            }
           ):
    """
    Run LLM with search engine :)
    """
    if url_fetch_support:
        new_messages = copy.deepcopy(messages)
        for m in new_messages:
            m["content"] = prepare_prompt(m["content"])
        messages = copy.deepcopy(new_messages)
    try:
        response: ChatResponse = chat(model=model,
                                      messages=messages,
                                      think=think,
                                      options=options,
                                      )
    except:
        print("\nUnavailable model...\n\n")
        return "", None
    # From source https://github.com/ollama/ollama-python
    # print(response['message']['content'])
    # # or access fields directly from the response object
    # print(response.message.content)
    metrics = extract_metrics(response)
    return response.message.content, metrics
