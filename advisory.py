import json 
with open("textdataset/diseases.json") as f:
    data = json.load(f)

def get_advisory(disease_id):
    for crop in data["crops"]:
        for disease in crop["diseases"]:
            if disease["disease_id"]==disease_id:
                return disease
    return None

prediction = "tomato_early_blight"

info = get_advisory(prediction)

print(info["symptoms"])
print(info["treatment"])
print(info["prevention"])

# print(info)