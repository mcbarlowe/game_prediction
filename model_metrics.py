import os
import datetime
import pandas as pd
import sys
from sklearn.metrics import accuracy_score, log_loss
from sqlalchemy import create_engine
from twython import Twython
from prediction_model import get_twitter_keys

def main():

    date = datetime.datetime.now().strftime('%Y-%m-%d')
    sql_string = 'select sched.game_id, predict.home_win_probs, predict.home_win_pred, sched.home_win from nhl_tables.predictions predict left join nhl_tables.nhl_schedule sched on predict.game_id = sched.game_id where sched.season=20182019 and sched.game_id > 2018020000;'


    engine = create_engine(os.environ.get('DEV_DB_CONNECT'))

    df = pd.read_sql(sql_string, con=engine)

    accuracy = accuracy_score(df.home_win, df.home_win_pred)

    logloss = log_loss(df.home_win, df.home_win_probs)

    accuracy = round(accuracy, 4)
    logloss = round(logloss, 4)

    print(f'Model Accuracy: {accuracy}')
    print(f'Model Logloss: {logloss}')

    twitter_keys = get_twitter_keys(sys.argv[1])

    #set twitter API
    twitter = Twython(twitter_keys['Consumer Key'], twitter_keys['Consumer Secret Key'],
                      twitter_keys['Access Key'], twitter_keys['Access Secret Key'])

    tweet_string = f'Game Prediction Model Metrics as of {date}\nAccuracy: {accuracy}\nLogloss {logloss}:\n'

    twitter.update_status(status=tweet_string)

    return

if __name__ == '__main__':
    main()
