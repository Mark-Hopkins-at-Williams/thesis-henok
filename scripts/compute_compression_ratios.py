import json

# experiment = 117
experiment = 103

with open(
    f"/mnt/storage/swexler/thesis-wexler/models/french-training-v{experiment}/experiment.json"
) as reader:
    config = json.load(reader)
test_set = config["corpora"]["fr-eng"]["eng"]["test"]
print(test_set)
uncompressed_test_set = "/mnt/storage/swexler/thesis-wexler/examples/french-data-7-mil-512-filtered/test.eng"

compressed_bytes = 0
with open(test_set) as test:
    for line in test:
        # bytes = line.strip().encode("utf-8")
        i = 0
        while i < len(line):
            if line[i] == "\\":
                compressed_bytes += 1
                i += 4
            else:
                compressed_bytes += 1
                i += 1


uncompressed_bytes = 0
with open(uncompressed_test_set) as test:
    for line in test:
        bytes = line.strip().encode("utf-8")
        uncompressed_bytes += len(bytes)

print(compressed_bytes)
print(uncompressed_bytes)

print(compressed_bytes / uncompressed_bytes)
