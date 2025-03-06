import json
import numpy as np


with open("../data/rawData.json", 'r', encoding='utf-8') as f:
    data = json.load(f)


transcriptions = [item['transcription'] for item in data.values() if item['language'] == 'Chinese']


transcriptions_array = np.array(transcriptions)

np.save('../data/data.npy', transcriptions_array)