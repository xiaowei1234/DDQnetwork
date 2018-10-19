#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 15:12:54 2018

@author: xiaowei
"""

from pymongo import MongoClient
import sys
import bson.json_util as json_util
import constants as c

# ssh -L 9998:10.128.1.43:27017 xwei@10.128.1.27
client = MongoClient('127.0.0.1', 9998)
db = client.billfloat


filt = {
        '$and':[
#                  {'$or': [
#                  {'decision_engine.reply.handset_amount_to_collect': {'$lt': c.max_amount}}
#                          ]}
                {'run_time_ms': {'$gt': 1535820020000, '$lt': 1538412020000}}
                , {'$or': [
                        {'decision_engine.reply.action': {'$in': [0, 1, 2, 3]}}
#                        , {'$and': [{c.action_col: c.zero_action}
#                                , {'decision_engine.reply.handset_amount_to_collect': {'$lt': 0.01}}
#                        ]}
                        , {'$and': [{'decision_engine.reply.handset_amount_to_collect': {'$lt': 250, '$gt': 5}}
                                , {'request.last_payback_failed_code':
                                       {'$nin': list(c.success_codes)}}
                        ]}
                    ]}
                , {'request.order_number': {'$exists': True}}
               ]}

    
col_filt = { 'decision_engine': 1, 'request': 1
           , 'run_time': 1, 'run_time_ms': 1 
           }


c2 = (db.collection_engine.find(filt, col_filt)
#       .sort('$natural', pymongo.DESCENDING)
     .limit(10)
     )

astr = json_util.dumps([j for j in c2])

sys.stdout.write(astr)
sys.stdout.flush()