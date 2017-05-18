namespace cpp suggestion

struct Result{
      1:double timeUsed,
      2:string sEngineTimeInfo,
      3:list<string> listWords,
}

service Suggestion { 
	Result getPrediction(1:string sWord,
                               2:string sLocale,
                               3:string sAppName)
}
