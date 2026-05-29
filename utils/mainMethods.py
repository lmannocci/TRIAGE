import sys
from telegram_send_message import telegram_send as t
from utils.log_manager.log_manager import *
from datetime import datetime
import torch

lm = LogManager("main")

def select_sources_ES():
    tweet_keys = ["created_at", "id_str", "user.id_str"]

    hashtag_keys = ["entities.hashtags.text",
                    "extended_tweet.entities.hashtags.text",  # .text
                    "retweeted_status.entities.hashtags.text",  # .text
                    "retweeted_status.extended_tweet.entities.hashtags.text"]  # .text

    mention_keys = ["entities.user_mentions.id",  # .id
                    "extended_tweet.entities.user_mentions.id",  # .id
                    "retweeted_status.entities.user_mentions.id",  # .id
                    "retweeted_status.extended_tweet.entities.user_mentions.id"]  # .id

    url_keys = ["entities.urls.expanded_url",  # .expanded_url
                "extended_tweet.entities.urls.expanded_url",  # .expanded_url
                "retweeted_status.entities.urls.expanded_url",  # .expanded_url
                "retweeted_status.extended_tweet.entities.urls.expanded_url"]  # expanded_url

    retweet_keys = ["retweeted_status.id_str",
                    "retweeted_status.created_at",
                    "retweeted_status.user.id_str"]

    reply_keys = ["in_reply_to_status_id_str",
                  "in_reply_to_user_id_str"]

    source_list = []
    source_list += tweet_keys
    source_list += retweet_keys
    source_list += reply_keys
    source_list += hashtag_keys
    source_list += mention_keys
    source_list += url_keys

    return source_list

def select_sources_info_tweet_ES():
    source_list = ["id_str", "user.id_str", "text", "favorite_count", "quote_count", "reply_count", "retweet_count", "created_at"]

    return source_list

def select_sources_info_user_ES():
    source_list = ["id_str", 'name', 'screen_name', 'description', 'location', "favourites_count", "followers_count", "friends_count", 'statuses_count', "botometer", "created_at"]

    return source_list

def get_formatted_date(date):
    input_date = datetime.strptime(date, "%Y-%m-%d %H:%M:%S")
    output_date_string = input_date.strftime("%a %b %d %H:%M:%S +0000 %Y")
    return output_date_string

def send_log():
    path_log = f".{os.sep}..{os.sep}logCBPlusPlus.txt"
    path_main_log = f".{os.sep}utils{os.sep}LogManager{os.sep}log{os.sep}log_main.txt"


    if os.path.exists(path_main_log):
        try:
            t.send_document(path_main_log)
        except Exception as error:
            lm.printl("Error. Impossible sending log_main.txt.")
            #lm.printl(error)
    if os.path.exists(path_log):
        try:
            t.send_document(path_log)
        except Exception as error:
            lm.printl("Error. Impossible sending logCBPlusPlus.txt.")
            # lm.printl(error)

def redefine_exception():
    # Set your custom excepthook
    sys.excepthook = custom_excepthook

def custom_excepthook(exc_type, exc_value, traceback):
    # lm.printl("Custom excepthook invoked:")
    # lm.printl(f"Exception Type: {exc_type}")
    lm.printl(f"Exception Value: {exc_value}")
    # Add your custom handling logic here


def create_directory(file_name, path):
    if not os.path.exists(path):
        os.mkdir(path)
        os.chmod(path, 0o777)
        lm.printl(f"{file_name}. Created directory {path}")


def get_algorithm_param(algorithm, tuple):
    if algorithm == "clique_percolation":
        return {"k": tuple[0], 'm': tuple[1]}
    elif algorithm == "abacus":
        return {"min_actors": tuple[0], 'min_layers': tuple[1]}
    elif algorithm == "glouvain":
        return {"omega": tuple[0], 'gamma': tuple[1]}
    elif algorithm in ["louvain", 'flat_ec_louvain', 'flat_nw_louvain', 'flat_weighted_sum_louvain', 'flat_weighted_average_louvain', 'flat_and_weighted_sum_louvain']:
        return {"resolution": tuple[0]}
    else:
        return {}

def available_GPUs():
    lm.printl("Available GPUs:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        lm.printl(f"GPU {i}: {torch.cuda.get_device_name(i)}")