import codecs
with codecs.open('data/news.en.heldout-00001-of-00050','r','utf-8') as f:
    total = 0
    k = 0
    ll = []
    for line in f:
        ll += ["aa"]
        total += len(line.strip().split())
        k += 1 
    print ll
    print total
    print k
