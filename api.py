import requests
while True:
    web=requests.get("http://dreamli.ltd/test.json")
    enfc=web.json()
    print(enfc["dream"])