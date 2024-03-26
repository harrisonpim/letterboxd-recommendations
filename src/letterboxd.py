from typing import List

import httpx
import scrapy


def scrape_watchlist(username: str) -> List[str]:
    """
    Fetch the list of films from a user's watchlist.

    :param str username: The Letterboxd username to scrape.
    :return List[str]: A list of Letterboxd film slugs from the user's watchlist.
    """
    first_page_url = f"https://letterboxd.com/{username}/watchlist/page/1/"
    response = httpx.get(first_page_url)
    response.raise_for_status()
    watchlist = []
    html = scrapy.Selector(text=response.text)
    print(html)
    films_on_this_page = html.css("div.film-poster").xpath("@data-film-slug").getall()
    watchlist.extend(films_on_this_page)

    i = 2
    while True:
        next_page_url = first_page_url.replace("/page/1/", f"/page/{i}/")
        response = httpx.get(next_page_url)
        response.raise_for_status()
        html = scrapy.Selector(text=response.text)
        films_on_this_page = (
            html.css("div.film-poster").xpath("@data-film-slug").getall()
        )
        if len(films_on_this_page) == 0:
            break
        watchlist.extend(films_on_this_page)
        i += 1

    return watchlist
