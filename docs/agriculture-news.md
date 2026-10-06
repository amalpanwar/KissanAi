# Agriculture news

The sidebar displays the five newest dated links from Down To Earth Hindi's
[agriculture section](https://hindi.downtoearth.org.in/agriculture), with publication
time and last successful fetch time in IST. A Streamlit fragment refreshes the
view every 15 minutes while the app is open; page visits and explicit news queries
also refresh expired data. This is an on-demand feed, not a background job when
nobody is using the app. It covers national/international reporting, not necessarily
the selected village.

The public Quintype collection endpoint supplies headline, subheadline, canonical
article URL and publication timestamp. No API key or user question is sent to the
publisher. Only these metadata are retained; full articles are not ingested or
republished. Responses are bounded by a six-second request timeout and a 2 MB limit.

The `news` specialist answers questions such as “आज की कृषि खबरें”, “latest
agriculture news”, and “pyaz ki taza khabar” with dated, linked headlines. The
agronomy agent can request related crop news through the existing message bus;
this optional addition reads the shared cache without waiting on the publisher.
Existing weather, price and pesticide tools remain responsible for their own data.
News text is never inserted into model instructions or treated as a pesticide
recommendation, village forecast or live mandi quote.

A fixed set of Hindi/English/Hinglish crop aliases plus explicit topic terms
matches available stories. Only the last 30 days of the fetched 30-story list are
eligible for chat. This is not a full-site search or article-body question-answering
system. Missing matches are stated explicitly, rather than substituting unrelated
stories. News answers are excluded from the reviewed-feedback training dataset
because their facts are time-sensitive.

The cache `data/processed/agriculture_news_cache.json` is ignored by Git. A
15-minute TTL avoids repeated requests, a one-minute failure cooldown prevents
retry storms, and a labelled last-good cache can be shown for at most 24 hours.
After that, the app says news is unavailable. Read-only hosting still benefits from
the process memory cache. Publication dates and cache freshness are distinct; an
old article never gets relabelled as today's publication.
