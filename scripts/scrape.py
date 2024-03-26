from typing import Any, Generator, Union

import scrapy


class LetterboxdSpider(scrapy.Spider):
    name = "letterboxd"
    allowed_domains = ["letterboxd.com"]
    start_urls = ["https://letterboxd.com/members/popular/this/all-time/"]
    n_pages = 10
    target_user = None

    def __init__(self, target_user=None, *args, **kwargs):
        super(LetterboxdSpider, self).__init__(*args, **kwargs)
        self.target_user = target_user

    def parse(self, response) -> Generator[Any, Any, None]:
        if self.target_user:
            yield response.follow(
                f"https://letterboxd.com/{self.target_user}/films/ratings",
                callback=self.parse_user,
            )
        else:
            user_links = response.css("a.avatar::attr(href)").getall()
            for user_link in user_links:
                yield response.follow(
                    user_link + "films/ratings", callback=self.parse_user
                )

            # only follow the first n_pages pages
            next_page = response.css("a.next::attr(href)").get()
            if next_page is not None and self.n_pages > 0:
                self.n_pages -= 1
                yield response.follow(next_page, callback=self.parse)

    def parse_user(
        self, response
    ) -> Generator[dict[str, Union[str, float]] | Any, Any, None]:
        username = response.url.split("/")[3]
        ratings = response.css("li.poster-container")
        for rating in ratings:
            rating_string = rating.css("span.rating::text").get()
            if rating_string is None:
                continue
            yield {
                "username": username,
                "film-slug": rating.css("div.poster::attr(data-film-slug)").get(),
                "rating": self._stars_to_rating(rating_string),
            }

        # Extract the link to the next page
        next_page = response.css("a.next::attr(href)").get()
        if next_page is not None:
            yield response.follow(next_page, callback=self.parse_user)

    def _stars_to_rating(self, stars: str) -> float:
        """
        Takes a string of stars and returns the number of stars as a float between 0 and 5.

        :param str stars: The raw star rating from letterboxd, eg "★★★½"
        :return float: The parsed star rating, eg 3.5
        """
        if (len(stars) > 5) or (stars.count("½") > 1):
            raise ValueError(f"Invalid star rating: {stars}")
        return stars.count("★") + 0.5 * stars.count("½")
