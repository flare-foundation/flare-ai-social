import time
import structlog
import tweepy
import google.generativeai as genai

from flare_ai_social.settings import settings

logger = structlog.get_logger(__name__)
genai.configure(api_key=settings.gemini_api_key)

def setup_twitter_api() -> tweepy.API:
    """
    Initialize and return the Tweepy API client using credentials from settings.
    """
    auth = tweepy.OAuthHandler(settings.x_api_key, settings.x_api_secret)
    auth.set_access_token(settings.x_access_token, settings.x_access_secret)
    api = tweepy.API(auth, wait_on_rate_limit=True)
    return api

def handle_mention(api: tweepy.API, mention: tweepy.Status, model: genai.GenerativeModel) -> None:
    """
    Given a mention tweet and a GenerativeModel, generate a response and reply on X (Twitter).
    """
    # 1. Extract text from the mention
    user_text = mention.text

    # 2. Build a prompt for your tuned model
    prompt = (
        "You are an autonomous AI agent representing Flare. "
        "Respond politely and helpfully to the user's Tweet. "
        f"User Tweet: \"{user_text}\""
    )

    # 3. Generate a response
    result = model.generate_content(prompt)
    ai_reply = result.text.strip()

    # 4. Reply to the mention
    try:
        logger.info("Replying to mention", mention_id=mention.id, ai_reply=ai_reply)
        api.update_status(
            status=f"@{mention.user.screen_name} {ai_reply}",
            in_reply_to_status_id=mention.id,
            auto_populate_reply_metadata=True
        )
    except Exception as e:
        logger.error("Error replying to mention", error=str(e))

def process_mentions(api: tweepy.API, model: genai.GenerativeModel, since_id: int = None) -> int:
    """
    Retrieve recent mentions, filter them, and respond using handle_mention.
    Returns the updated since_id.
    """
    mentions = api.mentions_timeline(count=20, since_id=since_id, tweet_mode="extended")
    new_since_id = since_id

    for mention in reversed(mentions):
        new_since_id = max(new_since_id or 0, mention.id)
        handle_mention(api, mention, model)

    return new_since_id

def start_social(tuned_model_id: str = "pugo-hillion") -> None:
    """
    Main entry point for our social AI agent:
    - Fetch and configure the tuned model
    - Initialize Twitter API
    - Retrieve mentions and reply in a loop
    """
    # 1. Print available tuned models
    tuned_models = [m.name for m in genai.list_tuned_models()]
    logger.info("available tuned models", tuned_models=tuned_models)

    # 2. Retrieve the tuned model
    model_info = genai.get_tuned_model(name=f"tunedModels/{tuned_model_id}")
    logger.info("tuned model info", model_info=model_info)
    model = genai.GenerativeModel(model_name=f"tunedModels/{tuned_model_id}")

    # 3. Setup Twitter API
    api = setup_twitter_api()

    # 4. Loop to process mentions
    since_id = None
    logger.info("Starting mention reply loop...")

    while True:
        since_id = process_mentions(api, model, since_id=since_id)
        time.sleep(30)  # Wait 30 seconds before checking mentions again

if __name__ == "__main__":
    start_social()
