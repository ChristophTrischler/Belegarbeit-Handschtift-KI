import json


def loadConfig(rows):
    return [row.replace("\n", "") for row in open(f"src/configs/config{rows}.txt")]


def loadRes():
    res = {}
    with open("src/res.json", "r") as jsonF:
        res = json.load(jsonF)
    return res


def compareTest(data):
    config = loadConfig(len(data))
    jsonData = {}
    with open("src/res.json", "r") as jsonF:
        jsonData = json.load(jsonF)
        for i in range(1, len(data)):
            charLoaded = config[i-1]
            for char in data[i]:
                jsonData[charLoaded]["all"] += 1
                if charLoaded.__contains__(char):
                    jsonData[charLoaded]["right"] += 1

            jsonData[charLoaded]["percentage"] = jsonData[charLoaded]["right"] / jsonData[charLoaded]["all"]

    print(jsonData)

    with open("src/res.json", "w") as jsonF:
        json.dump(jsonData, jsonF)
