# -*- coding: utf-8 -*-
"""
:Author: Jaekyoung Kim
:Date: 2017. 12. 20.
"""
import requests
import json

API_TOKEN = '461771069:AAGeDfAVHlf8tp5v46pzDB_24waE_F7K7uQ'
CHAT_ID = "382755174"
URL = "https://api.telegram.org/bot%s/sendMessage?chat_id=%s&text=%s"


def send_message(text):
    r = requests.get(URL % (API_TOKEN, CHAT_ID, text))
    json.loads(r.text)
