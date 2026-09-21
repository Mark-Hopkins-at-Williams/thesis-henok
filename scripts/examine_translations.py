import json

BASE_DIR = "experiments"
EXP_DIR = "enciphered_spa_to_spa-v6"


with open(f"{BASE_DIR}/{EXP_DIR}/translations.json") as reader:
    data1 = json.load(reader)
with open(f"{BASE_DIR}/{EXP_DIR}/references.json") as reader:
    data2 = json.load(reader)


for i in range(7):
    print(f"\nSENTENCE {i}")
    print("----------\n")
    for key in data1:
        print(f"{key}:")
        print(f"  sys  : {data1[key][i]}")
        print(f"  ref  : {data2[key][i]}")
