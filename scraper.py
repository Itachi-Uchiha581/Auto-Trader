import requests
from bs4 import BeautifulSoup


def scrape_blog(url):
    # Send a GET request to the UR
    my_headers = {
        "User-Agent": """Mozilla/5.0 (Macintosh; Intel Mac OSX 10_14_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/71.0.3578.98 Safari/537.36"""
    }
    session = requests.Session()
    response = session.get(url, headers=my_headers)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Parse the HTML content of the page using BeautifulSoup
        if "zacks" in url:
            # Find all paragraph tags and extract the text
            soup = BeautifulSoup(response.content, "html.parser")

            # Find all div elements with class "commentary_body" and extract the text
            commentary_divs = soup.find_all("div", class_="commentary_body")

            # Concatenate the text from all commentary divs
            all_text = " ".join(
                [commentary_div.get_text() for commentary_div in commentary_divs]
            )

            return all_text
        else:
            soup = BeautifulSoup(response.content, "html.parser")
            # Find and extract the text content of the blog post
            blog_text = soup.get_text()

            # Print or return the scraped text
            return blog_text
    else:
        print(f"Failed to retrieve the webpage. Status code: {response.status_code}")


# Example usage
# blog_url = "https://www.benzinga.com/news/24/02/36937329/whats-going-on-with-taiwan-semiconductor-manufacturing-stock-monday"
# scraped_text = scrape_blog(blog_url)

# Print or process the scraped text as needed
# print(scraped_text)
