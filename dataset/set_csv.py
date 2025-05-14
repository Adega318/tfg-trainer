import csv
import random

with open("Nos_Celtia-GL/metadata.csv", newline="", encoding="utf-8") as f:
    lector = list(csv.reader(f, delimiter="|"))

    header = ["audio_file", "text", "speaker_name"]

    limpias = []
    for row in lector:
        # Remove '.wav' from the first column
        row[0] = "wavs/" + row[0]
        limpias.append(row)

    random.shuffle(limpias)

    mid = int(0.8 * len(limpias) // 2)

    train = [row + ["@X"] for row in limpias[:mid]]
    evalue = [row + ["@X"] for row in limpias[mid:]]

    with open(
        "Nos_Celtia-GL/metadata_train.csv", "w", newline="", encoding="utf-8"
    ) as f_train:
        writer = csv.writer(f_train, delimiter="|")
        writer.writerow(header)
        writer.writerows(train)

    with open(
        "Nos_Celtia-GL/metadata_eval.csv", "w", newline="", encoding="utf-8"
    ) as f_test:
        writer = csv.writer(f_test, delimiter="|")
        writer.writerow(header)
        writer.writerows(evalue)
