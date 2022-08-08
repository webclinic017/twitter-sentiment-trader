import twitter
from .secrets import TWITTER_ACCESS, TWITTER_ACCESS_SECRET, TWITTER_CONSUMER, TWITTER_CONSUMER_SECRET

twitterAPI = twitter.Api(
    consumer_key=TWITTER_CONSUMER,
    consumer_secret=TWITTER_CONSUMER_SECRET,
    access_token_key=TWITTER_ACCESS,
    access_token_secret=TWITTER_ACCESS_SECRET
)
