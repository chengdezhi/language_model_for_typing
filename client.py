# -*- coding: utf-8 -*-
import traceback
import time
import sys
import thriftpy
import os
from thriftpy.transport import TFramedTransportFactory
from thriftpy.transport import TBufferedTransportFactory

from thriftpy.rpc import make_client

currentModulePath = os.path.split(os.path.realpath(__file__))[0] 

interface_thrift = thriftpy.load("interface.thrift",
                                module_name="interface_thrift")

class PredictClient:
    def __init__(self, ip="10.60.118.158", port=9888):
        self.ip = ip
        self.port = port
        try:
            self.client = make_client(interface_thrift.Suggestion, self.ip, self.port,
                                      trans_factory=TBufferedTransportFactory(), timeout=100000)
                                      #trans_factory=TFramedTransportFactory(), timeout=10000)
            print(self.client)
        except Exception as  err:
            print err
            traceback.print_exc()

hd = PredictClient()

try:
    result = hd.client.getPrediction("as soon as p", "en_US", "panda.com")
    print(result)
except Exception as err:
    print err
