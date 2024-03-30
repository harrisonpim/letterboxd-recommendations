from typing import List, Optional

import httpx
import scrapy


def scrape_watchlist(username: str, n_pages: Optional[int] = None) -> List[str]:
    """
    Fetch the list of films from a user's watchlist.

    :param str username: The Letterboxd username to scrape.
    :param int n_pages: The number of pages of the watchlist to scrape. If None, all
    pages will be scraped.
    :return List[str]: A list of Letterboxd film slugs from the user's watchlist.
    """
    first_page_url = f"https://letterboxd.com/{username}/watchlist/page/1/"
    response = httpx.get(first_page_url)
    response.raise_for_status()
    watchlist = []
    html = scrapy.Selector(text=response.text)
    films_on_this_page = html.css("div.film-poster").xpath("@data-film-slug").getall()
    watchlist.extend(films_on_this_page)

    if n_pages is None:
        n_pages = float("inf")

    if n_pages > 1:
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


def stars_to_rating(stars: str) -> float:
    """
    Takes a string of stars and returns the number of stars as a float between 0 and 5.

    :param str stars: The raw star rating from letterboxd, eg "★★★½"
    :return float: The parsed star rating, eg 3.5
    """
    if (len(stars) > 5) or (stars.count("½") > 1):
        raise ValueError(f"Invalid star rating: {stars}")
    return stars.count("★") + 0.5 * stars.count("½")
