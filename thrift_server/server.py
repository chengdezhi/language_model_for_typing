# -*- coding: utf-8 -*-
from api import  ngramPredict
import thriftpy
from thriftpy.rpc import make_server
interface_thrift = thriftpy.load("interface.thrift",
                                module_name="interface_thrift")


class PredictHandler(object):
    def __init__(self):
        self.log = {}
        self.lstmClient = ngramPredict() 

    def getPrediction(self,sWord,sLocale,sAppName):
        result = interface_thrift.Result()
        
        #TODO: ADD LSTM PREDICT
        result.timeUsed = 0.001
        result.sEngineTimeInfo = "1:0,3:0"
        
        result.listWords = self.lstmClient.get(sWord)
        return result


def main():
    ip = "0"
    port = 9090
    server = make_server(interface_thrift.Suggestion, PredictHandler(),
                         ip, port)
    print("serving...",ip,port)
    server.serve()


if __name__ == '__main__':
    main()

