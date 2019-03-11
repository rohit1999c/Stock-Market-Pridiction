import re
import requests
import sys
from pdb import set_trace as pb

def getCSV(keyword):

    symbol = keyword
    start_date = '1520764641' # start date timestamp
    end_date = '1552300641' # end date timestamp

    print "Inside process CSV file "
    print symbol
    print start_date
    print end_date

    crumble_link = 'https://finance.yahoo.com/quote/{0}/history?p={0}'
    crumble_regex = r'CrumbStore":{"crumb":"(.*?)"}'
    cookie_regex = r'set-cookie: (.*?);'
    quote_link = 'https://query1.finance.yahoo.com/v7/finance/download/{}?period1={}&period2={}&interval=1d&events=history&crumb={}'

    link = crumble_link.format(symbol)
    session = requests.Session()
    response = session.get(link)

    # get crumbs

    text = str(response.content)
    match = re.search(crumble_regex, text)
    crumbs = match.group(1)

    # get cookie

    cookie = session.cookies.get_dict()

    url = "https://query1.finance.yahoo.com/v7/finance/download/%s?period1=%s&period2=%s&interval=1d&events=history&crumb=%s" % (symbol, start_date, end_date, crumbs)

    r = requests.get(url,cookies=session.cookies.get_dict(),timeout=5, stream=True)

    out = r.text

    filename = '{}.csv'.format(symbol)

    with open(filename,'w') as f:
        f.write(out)
    print "filename ", filename